import json, numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2
'''
Gating and data association for multiple simultaneous targets.

P - predicted state covariance
H - measurement matrix
R - measurement noise covariance

Gating  — per-sensor, per-track Mahalanobis distance gate at P_G = 0.99
GNN     — one Hungarian assignment across all sensors simultaneously
'''


def ospa(estimated_positions, true_positions, c=100.0, p=2):
    """OSPA metric. c=cutoff (m), p=order. Inputs are lists of (2,) position arrays."""
    X = np.array(estimated_positions)
    Y = np.array(true_positions)
    n, m = len(X), len(Y)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return c
    D = np.minimum(np.linalg.norm(X[:, None] - Y[None, :], axis=2), c)
    if n <= m:
        row_ind, col_ind = linear_sum_assignment(D)
        loc_cost = sum(D[r, c_] ** p for r, c_ in zip(row_ind, col_ind))
    else:
        row_ind, col_ind = linear_sum_assignment(D.T)
        loc_cost = sum(D[c_, r] ** p for r, c_ in zip(row_ind, col_ind))
    return ((loc_cost + c**p * abs(n - m)) / max(n, m)) ** (1.0 / p)


def mahalanobis_d2(meas, z_pred, S):
    y = (meas - z_pred).reshape(-1, 1)
    y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi  # wrap bearing innovation
    return (y.T @ np.linalg.solve(S, y)).item()


def gating(
        actual_meas: NDArray[np.float64],
        z_pred: NDArray[np.float64],
        S: NDArray[np.float64],
) -> list[NDArray[np.float64]]:
    gamma = chi2.ppf(0.99, df=2)
    return [m for m in actual_meas if mahalanobis_d2(m, z_pred, S) < gamma]


def check_gating_sensor(t, sensor_id, all_meas, meas_list, active_tids, gated_per_track):
    """
    Runs gating tests 1-3 for one sensor stream.
    Returns (n_detected, n_in_gate) counts to accumulate into detection rate.

    """
    n_detected = 0
    n_in_gate  = 0

    for tid in active_tids:
        gated_meas = gated_per_track[sensor_id][tid]

        n_true = sum(1 for z, m in zip(all_meas, meas_list)
                     if any(np.allclose(z, gz) for gz in gated_meas)
                     and not m["is_false_alarm"] and m["target_id"] == tid)
        n_fa   = sum(1 for z, m in zip(all_meas, meas_list)
                     if any(np.allclose(z, gz) for gz in gated_meas)
                     and m["is_false_alarm"])

        print(f"t={t:6.1f}s  [{sensor_id}] track {tid}  |  "
              f"gate size={len(gated_meas)}  true_det={n_true}  FA_leaked={n_fa}")

        # Test 1 — never more than 1 true detection per track per scan
        assert n_true <= 1, \
            f"t={t} [{sensor_id}] track {tid}: {n_true} true detections in gate, expected ≤1"

        target_detected = any(not m["is_false_alarm"] and m["target_id"] == tid
                              for m in meas_list)

        # Test 2 — accumulate detection rate pre-crossing
        if t < 40.0 and target_detected:
            n_detected += 1
            if n_true == 1:
                n_in_gate += 1

        # Test 3 — gate not empty during crossing
        if 48.0 <= t <= 72.0 and target_detected:
            n_detected += 1
            if n_true == 1:
                n_in_gate += 1

    return n_detected, n_in_gate


def gnn_hungarian(
    tracks_pred_meas_radar,
    tracks_pred_meas_camera,
    all_radar,
    all_camera,
    S_radar,
    S_camera,
    gate_prob=0.99
):
    """Simultaneous GNN across radar and camera. Returns assignments as
    (track_idx, global_meas_idx, cost, sensor_id, local_meas_idx)."""
    n_tracks = len(tracks_pred_meas_radar)
    n_radar  = len(all_radar)
    n_camera = len(all_camera)

    all_meas = np.vstack([all_radar, all_camera]) if n_camera > 0 else all_radar
    n_meas   = len(all_meas)

    gamma  = chi2.ppf(gate_prob, df=2)
    BIG    = 1e6
    C      = np.full((n_tracks, n_meas), BIG)
    gated  = {i: {"radar": [], "camera": []} for i in range(n_tracks)}

    for i in range(n_tracks):
        for j in range(n_meas):
            sensor_id = "radar" if j < n_radar else "camera"
            z_pred    = tracks_pred_meas_radar[i]  if j < n_radar else tracks_pred_meas_camera[i]
            S         = S_radar[i]                 if j < n_radar else S_camera[i]
            d2        = mahalanobis_d2(all_meas[j], z_pred, S)
            if d2 < gamma:
                C[i, j] = d2

    row_ind, col_ind  = linear_sum_assignment(C)
    assignments       = []
    unassigned_tracks = set(range(n_tracks))
    unassigned_meas   = set(range(n_meas))

    for i, j in zip(row_ind, col_ind):
        if C[i, j] < BIG:
            sensor_id = "radar" if j < n_radar else "camera"
            local_idx = j       if j < n_radar else j - n_radar
            assignments.append((i, j, C[i, j], sensor_id, local_idx))
            unassigned_tracks.discard(i)
            unassigned_meas.discard(j)

    return assignments, list(unassigned_tracks), list(unassigned_meas), C


class MultiTargetTracker:
    """Owns all EKF track instances and runs the full predict-gate-associate-update cycle."""

    def __init__(self, cfm):
        self.cfm        = cfm
        self.trackers   = {}
        self.prev_times = {}

    def add_track(self, tid, state, cov, config, t0):
        from T4_radar_camera_fusion import RadarCameraFusionTracker
        self.trackers[tid]   = RadarCameraFusionTracker(
            cfm=self.cfm, state=state, covariance=cov, config=config)
        self.prev_times[tid] = t0

    def get_state(self, tid):
        return self.trackers[tid].get_state()

    def step(self, t, radar_meas, camera_meas, gnss_meas=None):
        """
        One scan cycle: predict → gate → GNN → update.
        Returns result dict with assignments, gated measurements per sensor per track, etc.
        """
        if gnss_meas:
            g = gnss_meas[-1]
            self.cfm.update_vessel_position(np.array([g["north_m"], g["east_m"]]))

        all_radar  = np.array([[m["range_m"], m["bearing_rad"]] for m in radar_meas]) \
                     if radar_meas  else np.empty((0, 2))
        all_camera = np.array([[m["range_m"], m["bearing_rad"]] for m in camera_meas]) \
                     if camera_meas else np.empty((0, 2))

        active_tids             = []
        tracks_pred_meas_radar  = []
        tracks_pred_meas_camera = []
        S_radar                 = []
        S_camera                = []
        # gated_per_track[sensor_id][tid] = list of gated measurements
        gated_per_track         = {"radar": {}, "camera": {}}

        for tid in self.trackers:
            if t <= self.prev_times[tid]:
                continue

            active_tids.append(tid)
            dt = t - self.prev_times[tid]
            self.trackers[tid].predict(dt)
            self.prev_times[tid] = t

            P = self.trackers[tid].get_covariance()

            # radar
            z_pred_r = self.cfm.h(self.trackers[tid].x, "radar")
            H_r, R_r = self.cfm.H(self.trackers[tid].x, "radar"), self.cfm.R("radar")
            S_r = H_r @ P @ H_r.T + R_r
            tracks_pred_meas_radar.append(z_pred_r)
            S_radar.append(S_r)
            gated_per_track["radar"][tid] = gating(all_radar, z_pred_r, S_r)

            # camera
            z_pred_c = self.cfm.h(self.trackers[tid].x, "camera")
            H_c, R_c = self.cfm.H(self.trackers[tid].x, "camera"), self.cfm.R("camera")
            S_c = H_c @ P @ H_c.T + R_c
            tracks_pred_meas_camera.append(z_pred_c)
            S_camera.append(S_c)
            gated_per_track["camera"][tid] = gating(all_camera, z_pred_c, S_c)

        radar_available  = len(radar_meas)  > 0
        camera_available = len(camera_meas) > 0

        assignments, unassigned_tracks, unassigned_meas, _ = gnn_hungarian(
            tracks_pred_meas_radar, tracks_pred_meas_camera,
            all_radar  if radar_available  else np.empty((0, 2)),
            all_camera if camera_available else np.empty((0, 2)),
            S_radar, S_camera)

        for track_idx, _, _, sensor_id, local_idx in assignments:
            tid        = active_tids[track_idx]
            meas_array = all_radar if sensor_id == "radar" else all_camera
            self.trackers[tid].update(meas_array[local_idx], sensor_id=sensor_id)

        return {
            "assignments":       assignments,
            "unassigned_tracks": unassigned_tracks,
            "unassigned_meas":   unassigned_meas,
            "active_tids":       active_tids,
            "all_radar":         all_radar,
            "all_camera":        all_camera,
            "gated_per_track":   gated_per_track,
        }


def print_scan_summary(t, assignments, unassigned_tracks, unassigned_meas, active_tids, all_radar):
    print(f"\nt={t:.1f}s  assignments: {[(a[0], a[3], a[4]) for a in assignments]}")
    for track_idx in unassigned_tracks:
        print(f"t={t:.1f}s  MISSED DETECTION: track {active_tids[track_idx]}")
    n_radar = len(all_radar)
    for meas_idx in unassigned_meas:
        if meas_idx < n_radar:
            print(f"t={t:.1f}s  UNMATCHED (→ track initiation): radar idx={meas_idx}")
        else:
            print(f"t={t:.1f}s  UNMATCHED (→ track initiation): camera idx={meas_idx - n_radar}")


def check_association(t, assignments, unassigned_tracks, active_tids, radar_meas, camera_meas):
    """Tests 4-7: identity preservation and assignment completeness."""

    def get_meas(sensor_id, local_idx):
        return radar_meas[local_idx] if sensor_id == "radar" else camera_meas[local_idx]

    # Test 4 — pre-crossing: no identity swap
    if t < 40.0:
        for track_idx, _, _, sensor_id, local_idx in assignments:
            tid = active_tids[track_idx]
            m   = get_meas(sensor_id, local_idx)
            if not m["is_false_alarm"]:
                assert m["target_id"] == tid, \
                    f"t={t}: track {tid} assigned {sensor_id} meas from target {m['target_id']} — identity swap"

    # Test 5 — no identity swap in crossing window
    if 48.0 <= t <= 72.0:
        for track_idx, _, _, sensor_id, local_idx in assignments:
            tid = active_tids[track_idx]
            m   = get_meas(sensor_id, local_idx)
            if not m["is_false_alarm"]:
                assert m["target_id"] == tid, \
                    f"t={t}: track {tid} assigned {sensor_id} meas from target {m['target_id']} — identity swap in crossing"

    # Test 6 — undetected by all sensors → must not be assigned
    for tid in active_tids:
        in_radar  = any(not m["is_false_alarm"] and m["target_id"] == tid for m in radar_meas)
        in_camera = any(not m["is_false_alarm"] and m["target_id"] == tid for m in camera_meas)
        if not in_radar and not in_camera:
            assert tid not in [active_tids[a[0]] for a in assignments], \
                f"track {tid} assigned a measurement when not detected by any sensor"

    # Test 7 — true measurement exists → track must be assigned
    for track_idx in unassigned_tracks:
        tid            = active_tids[track_idx]
        true_in_radar  = any(not m["is_false_alarm"] and m["target_id"] == tid for m in radar_meas)
        true_in_camera = any(not m["is_false_alarm"] and m["target_id"] == tid for m in camera_meas)
        assert not (true_in_radar or true_in_camera), \
            f"t={t}: track {tid} unassigned but true measurement exists in scan"


def assert_end_of_run(mtt, target_ids, data,
                      radar_detected, radar_in_gate,
                      camera_detected, camera_in_gate,
                      ospa_crossing, ospa_post,
                      motp_distances, motp_matches):
    """End-of-run assertions: detection rates, RMSE, OSPA, MOTP."""

    radar_det_rate = radar_in_gate / radar_detected if radar_detected > 0 else 0.0
    print(f"\nRadar detection rate:  {radar_det_rate:.1%}  ({radar_in_gate}/{radar_detected})")
    assert radar_det_rate > 0.75, f"Radar detection rate {radar_det_rate:.1%} too low"

    if camera_detected > 0:
        camera_det_rate = camera_in_gate / camera_detected
        print(f"Camera detection rate: {camera_det_rate:.1%}  ({camera_in_gate}/{camera_detected})")
        assert camera_det_rate > 0.75, f"Camera detection rate {camera_det_rate:.1%} too low"

    print("\nFinal position RMSE per track:")
    for tid in target_ids:
        last_gt = data["ground_truth"][str(tid)][-1]
        gt_pos  = np.array([last_gt[1], last_gt[2]])
        rmse    = np.linalg.norm(mtt.get_state(tid)[:2] - gt_pos)
        print(f"  track {tid}: RMSE = {rmse:.2f} m")
        assert rmse < 20.0, f"track {tid} final RMSE {rmse:.2f} m > 20 m — track lost"

    if ospa_crossing:
        mean_ospa_crossing = float(np.mean(ospa_crossing))
        print(f"\nMean OSPA during crossing (48–72s): {mean_ospa_crossing:.2f} m")
        assert mean_ospa_crossing < 40.0, f"Mean OSPA in crossing {mean_ospa_crossing:.2f} m > 40 m"

    if ospa_post:
        mean_ospa_post = float(np.mean(ospa_post))
        print(f"Mean OSPA after crossing (>72s):   {mean_ospa_post:.2f} m")
        assert mean_ospa_post < 30.0, f"Mean OSPA after crossing {mean_ospa_post:.2f} m > 30 m"

    if motp_matches > 0:
        motp = float(np.sum(motp_distances) / motp_matches)
        print(f"\nMOTP (full run): {motp:.2f} m")
        assert motp < 15.0, f"MOTP {motp:.2f} m > 15 m — localisation too poor"


def test_gating_scenario_d(json_path: str = "harbour_sim_output/scenario_D.json") -> None:

    from T2_CoordinateFrameManager import CoordinateFrameManager
    from T3_single_sensor_tracker import EKFConfig

    with open(json_path) as f:
        data = json.load(f)

    cfm = CoordinateFrameManager(
        camera_offset=np.array([-80.0, 120.0]),
        radar_R=np.diag([5.0**2, np.deg2rad(0.3)**2]),
        camera_R=np.diag([8.0**2, np.deg2rad(0.15)**2]),
        ais_R=np.eye(2),
    )
    cfm.update_vessel_position(np.array([0.0, 0.0]))

    target_ids = [0, 1, 2, 3]
    mtt = MultiTargetTracker(cfm)
    for tid in target_ids:
        first_gt = data["ground_truth"][str(tid)][0]
        state    = np.array(first_gt[1:], dtype=float)
        cov      = np.diag([50.0, 50.0, 5.0, 5.0])
        mtt.add_track(tid, state, cov, EKFConfig(sigma_a=0.5), t0=first_gt[0])

    gt_lookup = {tid: sorted(data["ground_truth"][str(tid)], key=lambda r: r[0])
                 for tid in target_ids}

    def gt_pos_at(tid, t):
        records = gt_lookup[tid]
        times   = np.array([r[0] for r in records])
        idx     = int(np.argmin(np.abs(times - t)))
        return np.array([records[idx][1], records[idx][2]])

    scan_times              = sorted(set(m["time"] for m in data["measurements"]
                                        if m["sensor_id"] == "radar"))
    radar_detected          = 0
    radar_in_gate           = 0
    camera_detected         = 0
    camera_in_gate          = 0
    ospa_crossing           = []
    ospa_post               = []
    motp_distances          = []
    motp_matches            = 0

    for t in scan_times:
        radar_meas  = [m for m in data["measurements"] if m["sensor_id"] == "radar"  and m["time"] == t]
        camera_meas = [m for m in data["measurements"] if m["sensor_id"] == "camera" and m["time"] == t]
        gnss_meas   = [m for m in data["measurements"] if m["sensor_id"] == "gnss"   and m["time"] == t]

        result = mtt.step(t, radar_meas, camera_meas, gnss_meas)

        assignments       = result["assignments"]
        unassigned_tracks = result["unassigned_tracks"]
        unassigned_meas   = result["unassigned_meas"]
        active_tids       = result["active_tids"]
        all_radar         = result["all_radar"]
        all_camera        = result["all_camera"]
        gated_per_track   = result["gated_per_track"]

        # Tests 1-3: gating per sensor
        nd, ni = check_gating_sensor(t, "radar",  all_radar,  radar_meas,  active_tids, gated_per_track)
        radar_detected += nd;  radar_in_gate += ni

        if camera_meas:
            nd, ni = check_gating_sensor(t, "camera", all_camera, camera_meas, active_tids, gated_per_track)
            camera_detected += nd;  camera_in_gate += ni

        # Tests 4-7: association correctness
        print_scan_summary(t, assignments, unassigned_tracks, unassigned_meas, active_tids, all_radar)
        check_association(t, assignments, unassigned_tracks, active_tids, radar_meas, camera_meas)

        # per-scan OSPA + MOTP accumulation
        est_pos_scan = [mtt.get_state(tid)[:2] for tid in target_ids]
        ospa_t       = ospa(est_pos_scan, [gt_pos_at(tid, t) for tid in target_ids], c=100.0, p=2)
        print(f"t={t:6.1f}s  OSPA={ospa_t:.2f} m")

        for tid in active_tids:
            motp_distances.append(np.linalg.norm(mtt.get_state(tid)[:2] - gt_pos_at(tid, t)))
            motp_matches += 1

        if 48.0 <= t <= 72.0:
            ospa_crossing.append(ospa_t)
        elif t > 72.0:
            ospa_post.append(ospa_t)

    assert_end_of_run(mtt, target_ids, data,
                      radar_detected, radar_in_gate,
                      camera_detected, camera_in_gate,
                      ospa_crossing, ospa_post,
                      motp_distances, motp_matches)


if __name__ == "__main__":
    test_gating_scenario_d()
