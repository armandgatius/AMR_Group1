import json, numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2
'''
This is gating for every track, predicted measurement
Gating is done per sensor

P - predicted state covariance
H - mesurement matrix
R - measurement noise covariance

Gating — reduces the search space, rejects obvious non-candidates (FA and far-away measurements)

Data association (GNN) — from whatever survived the gate, picks the single best assignment
Test 5 - covers no identity swap at the crossing test
Test 7 and test 8 - covers all 4 tracks confirmed and maintained throughout test

No ais measurements in scenario D  
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


#standalone gating function - just for testing
def gating(
        actual_meas :NDArray[np.float64],  #shape(M,2)
        predicted_meas: NDArray[np.float64], #shape(N,2)
        P: NDArray[np.float64], #shape(n,n)
        R: NDArray[np.float64], #shape(2,2)
        H: NDArray[np.float64] #shape(2,n)
        ) -> list[NDArray[np.float64]]:

    gate = []
    S = H @ P @ H.T + R
    gamma = chi2.ppf(0.99, df=2)

    for meas in actual_meas:
        mahalanobis_dist = mahalanobis_d2(meas = meas, z_pred=predicted_meas, S = S)
        if mahalanobis_dist < gamma:
            gate.append(meas)
    return gate


# both gating and data association — simultaneous across all sensor streams
def gnn_hungarian(
    tracks_pred_meas_radar,   # list of predicted meas, one per track, for radar
    tracks_pred_meas_camera,  # list of predicted meas, one per track, for camera
    all_radar,                # shape (N_radar, 2)
    all_camera,               # shape (N_camera, 2)
    S_radar,                  # list of S matrices, one per track, for radar
    S_camera,                 # list of S matrices, one per track, for camera
    gate_prob=0.99
):
    n_tracks = len(tracks_pred_meas_radar)
    n_radar  = len(all_radar)
    n_camera = len(all_camera)

    # pool radar then camera into one measurement array
    if n_camera > 0:
        all_meas = np.vstack([all_radar, all_camera])
    else:
        all_meas = all_radar
    n_meas = len(all_meas)

    gamma = chi2.ppf(gate_prob, df=2)
    BIG = 1e6
    C = np.full((n_tracks, n_meas), BIG)

    for i in range(n_tracks):
        for j in range(n_meas):
            if j < n_radar:
                z_pred = tracks_pred_meas_radar[i]
                S      = S_radar[i]
            else:
                z_pred = tracks_pred_meas_camera[i]
                S      = S_camera[i]
            d2 = mahalanobis_d2(all_meas[j], z_pred, S)
            if d2 < gamma:
                C[i, j] = d2

    row_ind, col_ind = linear_sum_assignment(C)
    assignments = []
    unassigned_tracks = set(range(n_tracks))
    unassigned_measurements = set(range(n_meas))

    for i, j in zip(row_ind, col_ind):
        if C[i, j] < BIG:
            sensor_id = "radar" if j < n_radar else "camera"
            local_idx = j if j < n_radar else j - n_radar
            assignments.append((i, j, C[i, j], sensor_id, local_idx))
            unassigned_tracks.discard(i)
            unassigned_measurements.discard(j)

    return assignments, list(unassigned_tracks), list(unassigned_measurements), C



def test_gating_scenario_d(json_path: str = "harbour_sim_output/scenario_D.json")->None:

    from T2_CoordinateFrameManager import CoordinateFrameManager
    from T4_radar_camera_fusion import RadarCameraFusionTracker

    with open(json_path) as f:
        data = json.load(f)

    cfm = CoordinateFrameManager(
        camera_offset=np.array([-80.0, 120.0]),
        radar_R=np.diag([5.0**2, np.deg2rad(0.3)**2]),
        camera_R=np.diag([8.0**2, np.deg2rad(0.15)**2]),
        ais_R=np.eye(2),
    )

    cfm.update_vessel_position(np.array([0.0, 0.0]))

     # initialise one tracker per target from its first true radar detection
    target_ids = [0, 1, 2, 3]
    trackers = {}
    prev_times = {}
    for tid in target_ids:
        gt_records = data["ground_truth"][str(tid)]
        first_gt = gt_records[0]  # [t, N, E, vN, vE]
        state = np.array(first_gt[1:], dtype=float)  # [N, E, vN, vE]
        cov = np.diag([50.0, 50.0, 5.0, 5.0])
        from T3_single_sensor_tracker import EKFConfig
        config = EKFConfig(sigma_a=0.5)
        trackers[tid] = RadarCameraFusionTracker(cfm=cfm, state=state, covariance=cov, config=config)
        prev_times[tid] = first_gt[0]


    # ground truth lookup per target
    gt_lookup = {tid: sorted(data["ground_truth"][str(tid)], key=lambda r: r[0])
                 for tid in target_ids}

    #finds ground truth pos for target tid at closest time to t
    def gt_pos_at(tid, t):
        records = gt_lookup[tid]
        times = np.array([r[0] for r in records])
        idx = int(np.argmin(np.abs(times - t)))
        return np.array([records[idx][1], records[idx][2]])

    #get all unique radar scan times
    scan_times = sorted(set(m["time"] for m in data["measurements"] if m["sensor_id"] == "radar"))
    total_detected = 0
    total_in_gate = 0
    ospa_crossing = []
    ospa_post     = []
    motp_distances = []   # one entry per matched (track, target) pair per scan
    motp_matches   = 0
    for t in scan_times:
        active_tids = []
        tracks_pred_meas_radar = []
        tracks_pred_meas_camera = []
        S_radar = []
        S_camera = []
        
        radar_meas_per_scan = [
            m for m in data["measurements"]
            if m["sensor_id"] == "radar" and m["time"] == t
        ]
        camera_meas_per_scan = [
            m for m in data["measurements"]
            if m["sensor_id"] == "camera" and m["time"] == t
        ]
        gnss_meas_per_scan = [
            m for m in data["measurements"]
            if m["sensor_id"] == "ais" and m["time"] == t
        ]
        if gnss_meas_per_scan:
            # take the latest GNSS fix (own ship position)
            g = gnss_meas_per_scan[-1]
            cfm.update_vessel_position(np.array([g["north_m"], g["east_m"]]))

        all_radar  = np.array([[m["range_m"], m["bearing_rad"]] for m in radar_meas_per_scan])
        all_camera = np.array([[m["range_m"], m["bearing_rad"]] for m in camera_meas_per_scan]) \
                     if camera_meas_per_scan else np.empty((0, 2))

        for tid in target_ids:

            if t <= prev_times[tid]:
                continue

            active_tids.append(tid)
            dt = t - prev_times[tid]
            trackers[tid].predict(dt)
            prev_times[tid] = t

            P = trackers[tid].get_covariance()

            z_pred_radar = cfm.h(trackers[tid].x, "radar")
            tracks_pred_meas_radar.append(z_pred_radar)
            H_r = cfm.H(trackers[tid].x, "radar")
            R_r = cfm.R("radar")
            S_radar.append(H_r @ P @ H_r.T + R_r)
            gated_meas = gating(all_radar, z_pred_radar, P, R_r, H_r)

            z_pred_camera = cfm.h(trackers[tid].x, "camera")
            tracks_pred_meas_camera.append(z_pred_camera)
            H_c = cfm.H(trackers[tid].x, "camera")
            R_c = cfm.R("camera")
            S_camera.append(H_c @ P @ H_c.T + R_c)

            #count what passed the gate using ground truth labels
            n_true = sum(1 for z,m in zip(all_radar, radar_meas_per_scan) if any(np.allclose(z,gz) for gz in gated_meas)
                         and not m["is_false_alarm"] and m["target_id"] == tid)
            n_fa   = sum(1 for z, m in zip(all_radar, radar_meas_per_scan)
                         if any(np.allclose(z, gz) for gz in gated_meas)
                         and m["is_false_alarm"])  # counting for false alarms from all tracks

            print(f"t={t:6.1f}s  track {tid}  |  gate size={len(gated_meas)}  "
            f"true_det={n_true}  FA_leaked={n_fa}")

            # Test 1 — never more than 1 true detection per track per scan
            assert n_true <= 1, f"t={t} track {tid}: {n_true} true detections in gate, expected at most 1"

            # for test 2 later
            if t < 40.0:
                target_detected = any(not m["is_false_alarm"] and m["target_id"] == tid
                                    for m in radar_meas_per_scan)
                if target_detected:
                    total_detected += 1
                    if n_true == 1:
                        total_in_gate += 1
            # Test 3 — crossing window: gate must not be completely empty
            if 48.0 <= t <= 72.0:
                target_detected = any(not m["is_false_alarm"] and m["target_id"] == tid
                                    for m in radar_meas_per_scan)
                if target_detected:
                    total_detected += 1
                    if n_true == 1:
                        total_in_gate += 1

        # sensor availability flags — a sensor is available if it reported at this scan time
        radar_available  = len(radar_meas_per_scan) > 0
        camera_available = len(camera_meas_per_scan) > 0

        assignments, unassigned_tracks, unassigned_meas, _ = gnn_hungarian(
            tracks_pred_meas_radar, tracks_pred_meas_camera,
            all_radar if radar_available  else np.empty((0, 2)),
            all_camera if camera_available else np.empty((0, 2)),
            S_radar, S_camera)

        print(f"\nt={t:.1f}s  assignments: {[(a[0], a[3], a[4]) for a in assignments]}")
        print(f"t={t:.1f}s  unassigned tracks: {unassigned_tracks}")
        print(f"t={t:.1f}s  unassigned meas:   {unassigned_meas}\n")

        # helper: look up the source measurement dict for an assignment
        def get_assigned_meas(sensor_id, local_idx):
            return radar_meas_per_scan[local_idx] if sensor_id == "radar" \
                   else camera_meas_per_scan[local_idx]

        # Test 4 — pre-crossing: no identity swap across all sensors
        if t < 40.0:
            for track_idx, _, _, sensor_id, local_idx in assignments:
                tid = active_tids[track_idx]
                assigned_m = get_assigned_meas(sensor_id, local_idx)
                if not assigned_m["is_false_alarm"]:
                    assert assigned_m["target_id"] == tid, \
                        f"t={t}: track {tid} assigned {sensor_id} meas from target {assigned_m['target_id']} — identity swap"

        # Test 5 — no identity swap across full crossing window, all sensors
        if 48.0 <= t <= 72.0:
            for track_idx, _, _, sensor_id, local_idx in assignments:
                tid = active_tids[track_idx]
                assigned_m = get_assigned_meas(sensor_id, local_idx)
                if not assigned_m["is_false_alarm"]:
                    assert assigned_m["target_id"] == tid, \
                        f"t={t}: track {tid} assigned {sensor_id} meas from target {assigned_m['target_id']} — identity swap in crossing"

        # Test 6 — if target undetected by all sensors, must not be assigned anything
        for tid in active_tids:
            in_radar  = any(not m["is_false_alarm"] and m["target_id"] == tid for m in radar_meas_per_scan)
            in_camera = any(not m["is_false_alarm"] and m["target_id"] == tid for m in camera_meas_per_scan)
            if not in_radar and not in_camera:
                assigned_track_indices = [a[0] for a in assignments]
                assert tid not in assigned_track_indices, \
                    f"track {tid} was assigned a measurement when target was not detected by any sensor"

        # Test 7 — if true measurement exists in any sensor, track must get assigned
        for track_idx in unassigned_tracks:
            tid = active_tids[track_idx]
            true_in_radar  = any(not m["is_false_alarm"] and m["target_id"] == tid for m in radar_meas_per_scan)
            true_in_camera = any(not m["is_false_alarm"] and m["target_id"] == tid for m in camera_meas_per_scan)
            assert not (true_in_radar or true_in_camera), \
                f"t={t}: track {tid} is unassigned but its true measurement exists in scan"

        # update each tracker with its assigned measurement
        for track_idx, _, _, sensor_id, local_idx in assignments:
            tid = active_tids[track_idx]
            meas_array = all_radar if sensor_id == "radar" else all_camera
            trackers[tid].update(meas_array[local_idx], sensor_id=sensor_id)

        # per-scan OSPA + accumulate MOTP distances
        est_pos_scan  = [trackers[tid].get_state()[:2] for tid in target_ids]
        true_pos_scan = [gt_pos_at(tid, t) for tid in target_ids]
        ospa_t = ospa(est_pos_scan, true_pos_scan, c=100.0, p=2)
        print(f"t={t:6.1f}s  OSPA={ospa_t:.2f} m")

        for tid in active_tids:
            d = np.linalg.norm(trackers[tid].get_state()[:2] - gt_pos_at(tid, t))
            motp_distances.append(d)
            motp_matches += 1

        if 48.0 <= t <= 72.0:
            ospa_crossing.append(ospa_t)
        elif t > 72.0:
            ospa_post.append(ospa_t)


    det_rate = total_in_gate / total_detected if total_detected > 0 else 0.0
    print(f"\nPre-crossing and crossing detection rate: {det_rate:.1%}  ({total_in_gate}/{total_detected})")
    assert det_rate > 0.75, f"Detection rate {det_rate:.1%} too low — gate too tight or tracker diverging"

    #Test 8
    print("\nFinal position RMSE per track:")
    for tid in target_ids:
        # get ground truth state at t=120s
        gt_records = data["ground_truth"][str(tid)]
        last_gt = gt_records[-1]  # [t, N, E, vN, vE]
        gt_pos = np.array([last_gt[1], last_gt[2]])  # [N, E]

        # estimated position from tracker
        est_pos = trackers[tid].get_state()[:2]  # [N, E]

        rmse = np.linalg.norm(est_pos - gt_pos)
        print(f"  track {tid}: RMSE = {rmse:.2f} m")
        assert rmse < 20.0, f"track {tid} final position error {rmse:.2f} m > 20 m — track lost"

    # Test 9 — OSPA during crossing must stay below 40m on average
    if ospa_crossing:
        mean_ospa_crossing = float(np.mean(ospa_crossing))
        print(f"\nMean OSPA during crossing (48–72s): {mean_ospa_crossing:.2f} m")
        assert mean_ospa_crossing < 40.0, \
            f"Mean OSPA in crossing {mean_ospa_crossing:.2f} m > 40 m — tracks diverged during crossing"

    # Test 10 — OSPA after crossing must stay below 30m on average
    if ospa_post:
        mean_ospa_post = float(np.mean(ospa_post))
        print(f"Mean OSPA after crossing (>72s):   {mean_ospa_post:.2f} m")
        assert mean_ospa_post < 30.0, \
            f"Mean OSPA after crossing {mean_ospa_post:.2f} m > 30 m — tracks did not recover"

    # Test 11 — MOTP < 15 m over the full run
    if motp_matches > 0:
        motp = float(np.sum(motp_distances) / motp_matches)
        print(f"\nMOTP (full run): {motp:.2f} m")
        assert motp < 15.0, f"MOTP {motp:.2f} m > 15 m — localisation too poor"

if __name__ == "__main__":
    test_gating_scenario_d()
