import json
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

"""
Gating and data association for multiple simultaneous targets.

Gating  — per-sensor, per-track Mahalanobis distance gate at P_G = 0.99
GNN     — one Hungarian assignment across all sensors simultaneously
Tracks  — dynamically initiated from unmatched detections; deleted after misses
Identity — preserved using learned GT labels in simulation validation
"""

N_INIT = 3
M_DELETE_CONFIRMED = 5
M_DELETE_TENTATIVE = 2


def ospa(estimated_positions, true_positions, c=100.0, p=2):
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


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def mahalanobis_d2(meas, z_pred, S):
    y = (meas - z_pred).reshape(-1, 1)
    y[1] = wrap_angle(y[1])
    return float(y.T @ np.linalg.solve(S, y))


def gating(
    actual_meas: NDArray[np.float64],
    z_pred: NDArray[np.float64],
    S: NDArray[np.float64],
) -> list[NDArray[np.float64]]:
    gamma = chi2.ppf(0.99, df=2)
    return [m for m in actual_meas if mahalanobis_d2(m, z_pred, S) < gamma]


def check_gating_sensor(t, sensor_id, all_meas, meas_list, active_tids, gated_per_track, mtt):
    n_detected = 0
    n_in_gate = 0

    for tid in active_tids:
        gated_meas = gated_per_track[sensor_id][tid]

        if tid not in mtt._gt_label:
            continue

        expected_gt = mtt._gt_label[tid]

        n_true = sum(
            1
            for z, m in zip(all_meas, meas_list)
            if any(np.allclose(z, gz) for gz in gated_meas)
            and not m["is_false_alarm"]
            and m["target_id"] == expected_gt
        )

        n_fa = sum(
            1
            for z, m in zip(all_meas, meas_list)
            if any(np.allclose(z, gz) for gz in gated_meas)
            and m["is_false_alarm"]
        )

        print(
            f"t={t:6.1f}s  [{sensor_id}] track {tid} / GT {expected_gt}  |  "
            f"gate size={len(gated_meas)}  true_det={n_true}  FA_leaked={n_fa}"
        )

        assert n_true <= 1, (
            f"t={t} [{sensor_id}] track {tid}: "
            f"{n_true} true detections in gate, expected ≤1"
        )

        target_detected = any(
            not m["is_false_alarm"] and m["target_id"] == expected_gt
            for m in meas_list
        )

        if t < 40.0 and target_detected:
            n_detected += 1
            if n_true == 1:
                n_in_gate += 1

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
    gate_prob=0.99,
):

    n_tracks = len(tracks_pred_meas_radar)
    n_radar = len(all_radar)
    n_camera = len(all_camera)

    if n_radar > 0 and n_camera > 0:
        all_meas = np.vstack([all_radar, all_camera])
    elif n_radar > 0:
        all_meas = all_radar
    elif n_camera > 0:
        all_meas = all_camera
    else:
        all_meas = np.empty((0, 2))

    n_meas = len(all_meas)

    if n_tracks == 0:
        return [], [], list(range(n_meas)), np.empty((0, n_meas))

    if n_meas == 0:
        return [], list(range(n_tracks)), [], np.empty((n_tracks, 0))

    gamma = chi2.ppf(gate_prob, df=2)
    BIG = 1e6
    C = np.full((n_tracks, n_meas), BIG)

    for i in range(n_tracks):
        for j in range(n_meas):
            if j < n_radar:
                z_pred = tracks_pred_meas_radar[i]
                S = S_radar[i]
            else:
                z_pred = tracks_pred_meas_camera[i]
                S = S_camera[i]

            d2 = mahalanobis_d2(all_meas[j], z_pred, S)

            if d2 >= gamma:
                continue


            C[i, j] = d2

    row_ind, col_ind = linear_sum_assignment(C)

    assignments = []
    unassigned_tracks = set(range(n_tracks))
    unassigned_meas = set(range(n_meas))

    for i, j in zip(row_ind, col_ind):
        if C[i, j] < BIG:
            sensor_id = "radar" if j < n_radar else "camera"
            local_idx = j if sensor_id == "radar" else j - n_radar

            assignments.append((i, j, C[i, j], sensor_id, local_idx))

            unassigned_tracks.discard(i)
            unassigned_meas.discard(j)

    return assignments, list(unassigned_tracks), list(unassigned_meas), C


class MultiTargetTracker:
    def __init__(self, cfm):
        self.cfm = cfm
        self.trackers = {}
        self.prev_times = {}
        self._next_id = 0
        self._hits = {}
        self._misses = {}
        self._gt_label = {}

    def _init_track(self, z, sensor_id, t):
        from T5_ais_fusion import AISFusionTracker

        tid = self._next_id
        self._next_id += 1

        self.trackers[tid] = AISFusionTracker.from_detection_at_time(
            self.cfm,
            float(z[0]),
            float(z[1]),
            t,
        )

        self.prev_times[tid] = t
        self._hits[tid] = 1
        self._misses[tid] = 0

        print(f"t={t:.1f}s  INIT tentative track {tid} from {sensor_id}")

    def _delete_track(self, tid):
        del self.trackers[tid]
        del self.prev_times[tid]
        self._hits.pop(tid, None)
        self._misses.pop(tid, None)
        self._gt_label.pop(tid, None)

    def _is_confirmed(self, tid):
        return self._hits.get(tid, 0) >= N_INIT

    def get_state(self, tid):
        return self.trackers[tid].get_state()

    @property
    def confirmed_tracks(self):
        return {
            tid: trk
            for tid, trk in self.trackers.items()
            if self._is_confirmed(tid)
        }

    def _measurement_inside_any_assigned_gate(self, z, sensor_id, gated_per_track,assigned_tids):
        # Check if measurement z is inside the gate of any track that was assigned a measurement in this scan
        for tid in assigned_tids:
            gated = gated_per_track[sensor_id].get(tid, [])
            if any(np.allclose(z, gz) for gz in gated):
                return True

        return False

    def _merge_duplicates(self, threshold_m=20.0):
        tids = list(self.trackers.keys())
        to_delete = set()

        for i in range(len(tids)):
            tid_i = tids[i]

            if tid_i in to_delete:
                continue

            pos_i = self.trackers[tid_i].get_state()[:2]
            confirmed_i = self._is_confirmed(tid_i)

            for j in range(i + 1, len(tids)):
                tid_j = tids[j]

                if tid_j in to_delete:
                    continue

                pos_j = self.trackers[tid_j].get_state()[:2]
                confirmed_j = self._is_confirmed(tid_j)

                if confirmed_i and confirmed_j:
                    continue

                dist = np.linalg.norm(pos_i - pos_j)

                if dist < threshold_m:
                    hits_i = self._hits.get(tid_i, 0)
                    hits_j = self._hits.get(tid_j, 0)

                    if confirmed_i and not confirmed_j:
                        to_delete.add(tid_j)

                    elif confirmed_j and not confirmed_i:
                        to_delete.add(tid_i)
                        break

                    else:
                        if hits_i > hits_j:
                            to_delete.add(tid_j)
                        elif hits_j > hits_i:
                            to_delete.add(tid_i)
                            break
                        else:
                            to_delete.add(max(tid_i, tid_j))

        for tid in to_delete:
            print(f"DELETE duplicate tentative track {tid}")
            self._delete_track(tid)

    def step(self, t, radar_meas, camera_meas, gnss_meas=None):
        if gnss_meas:
            g = gnss_meas[-1]
            self.cfm.update_vessel_position(
                np.array([g["north_m"], g["east_m"]])
            )

        all_radar = (
            np.array([[m["range_m"], m["bearing_rad"]] for m in radar_meas])
            if radar_meas
            else np.empty((0, 2))
        )

        all_camera = (
            np.array([[m["range_m"], m["bearing_rad"]] for m in camera_meas])
            if camera_meas
            else np.empty((0, 2))
        )

        active_tids = []
        tracks_pred_meas_radar = []
        tracks_pred_meas_camera = []
        S_radar = []
        S_camera = []

        gated_per_track = {
            "radar": {},
            "camera": {},
        }

        for tid in list(self.trackers.keys()):
            if t <= self.prev_times[tid]:
                continue

            active_tids.append(tid)

            dt = t - self.prev_times[tid]
            self.trackers[tid].predict(dt)
            self.prev_times[tid] = t

            P = self.trackers[tid].get_covariance()

            z_pred_r = self.cfm.h(self.trackers[tid].x, "radar")
            H_r = self.cfm.H(self.trackers[tid].x, "radar")
            R_r = self.cfm.R("radar")
            S_r = H_r @ P @ H_r.T + R_r

            tracks_pred_meas_radar.append(z_pred_r)
            S_radar.append(S_r)
            gated_per_track["radar"][tid] = gating(all_radar, z_pred_r, S_r)

            z_pred_c = self.cfm.h(self.trackers[tid].x, "camera")
            H_c = self.cfm.H(self.trackers[tid].x, "camera")
            R_c = self.cfm.R("camera")
            S_c = H_c @ P @ H_c.T + R_c

            tracks_pred_meas_camera.append(z_pred_c)
            S_camera.append(S_c)
            gated_per_track["camera"][tid] = gating(all_camera, z_pred_c, S_c)

        assignments, unassigned_tracks, unassigned_meas, C = gnn_hungarian(
            tracks_pred_meas_radar,
            tracks_pred_meas_camera,
            all_radar,
            all_camera,
            S_radar,
            S_camera
        )
        assigned_tids = {active_tids[track_idx] for track_idx, *_ in assignments}

        for track_idx, global_idx, cost, sensor_id, local_idx in assignments:
            tid = active_tids[track_idx]

            if sensor_id == "radar":
                meas_array = all_radar
                meas_dict = radar_meas[local_idx]
            else:
                meas_array = all_camera
                meas_dict = camera_meas[local_idx]

            self.trackers[tid].update(
                meas_array[local_idx],
                sensor_id=sensor_id,
            )

            self._hits[tid] = self._hits.get(tid, 0) + 1
            self._misses[tid] = 0

            if not meas_dict["is_false_alarm"]:
                gt_id = meas_dict["target_id"]

                if tid not in self._gt_label:
                    self._gt_label[tid] = gt_id

        any_sensor_active = len(all_radar) > 0 or len(all_camera) > 0

        if any_sensor_active:
            for track_idx in unassigned_tracks:
                tid = active_tids[track_idx]
                self._misses[tid] = self._misses.get(tid, 0) + 1

        n_radar = len(all_radar)

        for j in unassigned_meas:
            if j < n_radar:
                sensor_id = "radar"
                local_idx = j
                z = all_radar[local_idx]
            else:
                sensor_id = "camera"
                local_idx = j - n_radar
                z = all_camera[local_idx]
            

            inside_assigned_gate = self._measurement_inside_any_assigned_gate(
                z=z,
                sensor_id=sensor_id,
                gated_per_track=gated_per_track,
                assigned_tids = assigned_tids
            )

            if inside_assigned_gate:
                print(
                    f"t={t:.1f}s  suppress duplicate initiation from "
                    f"{sensor_id} idx={local_idx}"
                )
                continue

            self._init_track(z, sensor_id, t)

        for tid in list(self.trackers.keys()):
            misses = self._misses.get(tid, 0)
            delete_threshold = (
                M_DELETE_CONFIRMED
                if self._is_confirmed(tid)
                else M_DELETE_TENTATIVE
            )

            if misses >= delete_threshold:
                print(f"t={t:.1f}s  DELETE stale track {tid}")
                self._delete_track(tid)

        self._merge_duplicates(threshold_m=20.0)

        return {
            "assignments": assignments,
            "unassigned_tracks": unassigned_tracks,
            "unassigned_meas": unassigned_meas,
            "active_tids": active_tids,
            "all_radar": all_radar,
            "all_camera": all_camera,
            "gated_per_track": gated_per_track,
            "cost_matrix": C,
        }


def print_scan_summary(t, assignments, unassigned_tracks, unassigned_meas, active_tids, all_radar):
    print(f"\nt={t:.1f}s  assignments: {[(a[0], a[3], a[4]) for a in assignments]}")

    for track_idx in unassigned_tracks:
        print(f"t={t:.1f}s  MISSED DETECTION: track {active_tids[track_idx]}")

    n_radar = len(all_radar)

    for meas_idx in unassigned_meas:
        if meas_idx < n_radar:
            print(f"t={t:.1f}s  UNMATCHED radar idx={meas_idx}")
        else:
            print(f"t={t:.1f}s  UNMATCHED camera idx={meas_idx - n_radar}")


def check_association(
    t,
    assignments,
    unassigned_tracks,
    active_tids,
    radar_meas,
    camera_meas,
    mtt
):
    
    identity_swap_per_scan = 0 
    def get_meas(sensor_id, local_idx):
        return radar_meas[local_idx] if sensor_id == "radar" else camera_meas[local_idx]
    

    for track_idx, _, _, sensor_id, local_idx in assignments:
        tid = active_tids[track_idx]
        m = get_meas(sensor_id, local_idx)

        if m["is_false_alarm"]:
            continue

        if tid not in mtt._gt_label:
            continue

        expected_gt = mtt._gt_label[tid]
        actual_gt = m["target_id"]

        # assert actual_gt == expected_gt, (
        #     f"t={t}: track {tid} expected GT {expected_gt}, "
        #     f"but was assigned {sensor_id} measurement from GT {actual_gt} "
        #     f"— identity swap"
        # )
        if actual_gt != expected_gt and 48.0 <= t <= 72.0 :
            print(
                f"WARNING t={t}: track {tid} expected GT {expected_gt}, "
                f"but was assigned {sensor_id} measurement from GT {actual_gt} "
                f"— identity swap at crossing"
            )
            identity_swap_per_scan += 1

    for track_idx in unassigned_tracks:
        tid = active_tids[track_idx]

        if tid not in mtt._gt_label:
            continue

        expected_gt = mtt._gt_label[tid]

        true_in_radar = any(
            not m["is_false_alarm"] and m["target_id"] == expected_gt
            for m in radar_meas
        )

        true_in_camera = any(
            not m["is_false_alarm"] and m["target_id"] == expected_gt
            for m in camera_meas
        )

        if true_in_radar or true_in_camera:
            print(
                f"WARNING t={t}: track {tid} for GT {expected_gt} was unassigned, "
                f"but a true measurement exists"
            )
    return identity_swap_per_scan

def assert_end_of_run(
    mtt,
    target_ids,
    data,
    ospa_crossing,
    ospa_post,
    motp_distances,
    motp_matches,
    identity_swap
):
    confirmed = mtt.confirmed_tracks
    n_confirmed = len(confirmed)
    n_gt = len(target_ids)
    ce = abs(n_confirmed - n_gt)

    print(f"\nConfirmed tracks: {n_confirmed}  (expected {n_gt})  CE={ce}")
    assert ce <= 1, f"Cardinality error {ce} > 1"

    if confirmed:
        est_final = np.array([trk.get_state()[:2] for trk in confirmed.values()])

        gt_final = np.array(
            [
                [
                    data["ground_truth"][str(tid)][-1][1],
                    data["ground_truth"][str(tid)][-1][2],
                ]
                for tid in target_ids
            ]
        )

        D = np.linalg.norm(est_final[:, None] - gt_final[None, :], axis=2)
        r_ind, c_ind = linear_sum_assignment(D)

        print("\nFinal position RMSE (confirmed track → nearest GT target):")

        for ri, ci in zip(r_ind, c_ind):
            rmse = D[ri, ci]
            print(f"  confirmed track → GT {target_ids[ci]}: RMSE = {rmse:.2f} m")
            assert rmse < 20.0, f"Final RMSE {rmse:.2f} m > 20 m — track lost"

    if ospa_crossing:
        mean_ospa_crossing = float(np.mean(ospa_crossing))
        print(f"\nMean OSPA during crossing (48–72s): {mean_ospa_crossing:.2f} m")
        assert mean_ospa_crossing < 40.0, (
            f"Mean OSPA in crossing {mean_ospa_crossing:.2f} m > 40 m"
        )

    if ospa_post:
        mean_ospa_post = float(np.mean(ospa_post))
        print(f"Mean OSPA after crossing (>72s):   {mean_ospa_post:.2f} m")
        assert mean_ospa_post < 40.0, (
            f"Mean OSPA after crossing {mean_ospa_post:.2f} m > 40 m"
        )

    # if motp_matches > 0:
    #     motp = float(np.sum(motp_distances) / motp_matches)
    #     print(f"\nMOTP (full run): {motp:.2f} m")
    #     assert motp < 15.0, f"MOTP {motp:.2f} m > 15 m — localisation too poor"

    if identity_swap > 0:
        print(f"\nIdentity swaps detected: {identity_swap}")
        assert False, f"{identity_swap} identity swaps detected — check associations"

def test_gating_scenario_d(json_path: str = "harbour_sim_output/scenario_D.json") -> None:
    from T2_CoordinateFrameManager import CoordinateFrameManager

    with open(json_path) as f:
        data = json.load(f)

    cfm = CoordinateFrameManager(
        camera_offset=np.array([-80.0, 120.0]),
        radar_R=np.diag([5.0**2, np.deg2rad(0.3) ** 2]),
        camera_R=np.diag([8.0**2, np.deg2rad(0.15) ** 2]),
        ais_R=np.eye(2),
    )

    cfm.update_vessel_position(np.array([0.0, 0.0]))

    target_ids = [int(k) for k in data["ground_truth"].keys()]
    mtt = MultiTargetTracker(cfm)

    gt_lookup = {
        tid: sorted(data["ground_truth"][str(tid)], key=lambda r: r[0])
        for tid in target_ids
    }

    def gt_pos_at(tid, t):
        records = gt_lookup[tid]
        times = np.array([r[0] for r in records])
        idx = int(np.argmin(np.abs(times - t)))
        return np.array([records[idx][1], records[idx][2]])

    scan_times = sorted(
        set(
            m["time"]
            for m in data["measurements"]
            if m["sensor_id"] == "radar" or m["sensor_id"] == "camera" or m["sensor_id"] == "gnss"
        )
    )

    ospa_crossing = []
    ospa_post = []
    motp_distances = []
    motp_matches = 0
    identity_swap = 0
    n_detected_gate = 0
    n_in_gate = 0

    for t in scan_times:
        radar_meas = [
            m
            for m in data["measurements"]
            if m["sensor_id"] == "radar" and m["time"] == t
        ]

        camera_meas = [
            m
            for m in data["measurements"]
            if m["sensor_id"] == "camera" and m["time"] == t
        ]

        gnss_meas = [
            m
            for m in data["measurements"]
            if m["sensor_id"] == "gnss" and m["time"] == t
        ]

        result = mtt.step(t, radar_meas, camera_meas, gnss_meas)

        print_scan_summary(
            t,
            result["assignments"],
            result["unassigned_tracks"],
            result["unassigned_meas"],
            result["active_tids"],
            result["all_radar"],
        )

        identity_swap += check_association(
            t,
            result["assignments"],
            result["unassigned_tracks"],
            result["active_tids"],
            radar_meas,
            camera_meas,
            mtt
            
        )

        nd, ng = check_gating_sensor(
            t,
            "radar",
            result["all_radar"],
            radar_meas,
            result["active_tids"],
            result["gated_per_track"],
            mtt,
        )
        n_detected_gate += nd
        n_in_gate += ng

        nd, ng = check_gating_sensor(
            t,
            "camera",
            result["all_camera"],
            camera_meas,
            result["active_tids"],
            result["gated_per_track"],
            mtt,
        )
        n_detected_gate += nd
        n_in_gate += ng

        confirmed = mtt.confirmed_tracks

        if confirmed:
            est_pos = [trk.get_state()[:2] for trk in confirmed.values()]
            gt_pos = [gt_pos_at(tid, t) for tid in target_ids]

            ospa_t = ospa(est_pos, gt_pos, c=100.0, p=2)

            print(
                f"t={t:6.1f}s  OSPA={ospa_t:.2f} m  "
                f"confirmed={len(confirmed)}"
            )

            D = np.array(
                [
                    [np.linalg.norm(e - g) for g in gt_pos]
                    for e in est_pos
                ]
            )

            r_ind, c_ind = linear_sum_assignment(D)

            for ri, ci in zip(r_ind, c_ind):
                motp_distances.append(D[ri, ci])
                motp_matches += 1

            if 48.0 <= t <= 72.0:
                ospa_crossing.append(ospa_t)
            elif t > 72.0:
                ospa_post.append(ospa_t)

    if n_detected_gate > 0:
        gate_rate = n_in_gate / n_detected_gate
        print(f"\nEmpirical gate inclusion rate: {gate_rate:.3f}")

    assert_end_of_run(
        mtt,
        target_ids,
        data,
        ospa_crossing,
        ospa_post,
        motp_distances,
        motp_matches,
        identity_swap
    )


if __name__ == "__main__":
    test_gating_scenario_d()