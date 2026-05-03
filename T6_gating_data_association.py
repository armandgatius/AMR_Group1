import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

"""
T6: Gating and Data Association

Responsibilities:
  - Predict all EKF tracks forward to scan time
  - Mahalanobis gating per track per sensor
  - GNN assignment across radar, camera, and AIS jointly
  - EKF update for matched tracks

NOT responsible for:
  - track initiation
  - deletion
  - confirmation
  - hit/miss counting
  - duplicate merging
"""


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


def mahalanobis_d2(meas, z_pred, S, wrap_bearing=True):
    y = (meas - z_pred).reshape(-1, 1)

    if wrap_bearing:
        y[1] = wrap_angle(y[1])

    return float(y.T @ np.linalg.solve(S, y))


def gating(
    actual_meas: NDArray[np.float64],
    z_pred: NDArray[np.float64],
    S: NDArray[np.float64],
    wrap_bearing=True,
) -> list[NDArray[np.float64]]:
    gamma = chi2.ppf(0.99, df=2)

    return [
        m
        for m in actual_meas
        if mahalanobis_d2(m, z_pred, S, wrap_bearing=wrap_bearing) < gamma
    ]


def gnn_hungarian(
    tracks_pred_meas_radar,
    tracks_pred_meas_camera,
    tracks_pred_meas_ais,
    all_radar,
    all_camera,
    all_ais,
    S_radar,
    S_camera,
    S_ais,
    gate_prob=0.99,
):
    n_tracks = len(tracks_pred_meas_radar)

    n_radar = len(all_radar)
    n_camera = len(all_camera)
    n_ais = len(all_ais)

    blocks = []

    if n_radar > 0:
        blocks.append(all_radar)

    if n_camera > 0:
        blocks.append(all_camera)

    if n_ais > 0:
        blocks.append(all_ais)

    all_meas = np.vstack(blocks) if blocks else np.empty((0, 2))
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
                wrap_bearing = True

            elif j < n_radar + n_camera:
                z_pred = tracks_pred_meas_camera[i]
                S = S_camera[i]
                wrap_bearing = True

            else:
                z_pred = tracks_pred_meas_ais[i]
                S = S_ais[i]
                wrap_bearing = False

            d2 = mahalanobis_d2(
                all_meas[j],
                z_pred,
                S,
                wrap_bearing=wrap_bearing,
            )

            if d2 < gamma:
                C[i, j] = d2

    row_ind, col_ind = linear_sum_assignment(C)

    assignments = []
    unassigned_tracks = set(range(n_tracks))
    unassigned_meas = set(range(n_meas))

    for i, j in zip(row_ind, col_ind):
        if C[i, j] < BIG:
            if j < n_radar:
                sensor_id = "radar"
                local_idx = j

            elif j < n_radar + n_camera:
                sensor_id = "camera"
                local_idx = j - n_radar

            else:
                sensor_id = "ais"
                local_idx = j - n_radar - n_camera

            assignments.append((i, j, C[i, j], sensor_id, local_idx))

            unassigned_tracks.discard(i)
            unassigned_meas.discard(j)

    return assignments, list(unassigned_tracks), list(unassigned_meas), C


class MultiTargetTracker:
    """
    Predict -> gate -> assign -> update.

    Lifecycle decisions live in T7_Track_management.TrackManager.
    """

    def __init__(self, cfm):
        self.cfm = cfm
        self.trackers = {}
        self.prev_times = {}
        self._next_id = 0
        self._gt_label = {}

    def _init_track(self, x0, P0, t):
        from T5_ais_fusion import AISFusionTracker

        tid = self._next_id
        self._next_id += 1

        tracker = AISFusionTracker(
            cfm=self.cfm,
            state=x0,
            covariance=P0,
        )

        tracker.last_time = t

        self.trackers[tid] = tracker
        self.prev_times[tid] = t

        print(f"t={t:.1f}s  INIT EKF track {tid}")

        return tid

    def get_state(self, tid):
        return self.trackers[tid].get_state()

    def step(self, t, radar_meas, camera_meas, gnss_meas=None, ais_meas=None):
        if gnss_meas:
            g = gnss_meas[-1]

            self.cfm.update_vessel_position(
                np.array(
                    [
                        g["north_m"],
                        g["east_m"],
                    ],
                    dtype=float,
                )
            )

        all_radar = (
            np.array(
                [
                    [
                        m["range_m"],
                        m["bearing_rad"],
                    ]
                    for m in radar_meas
                ],
                dtype=float,
            )
            if radar_meas
            else np.empty((0, 2))
        )

        all_camera = (
            np.array(
                [
                    [
                        m["range_m"],
                        m["bearing_rad"],
                    ]
                    for m in camera_meas
                ],
                dtype=float,
            )
            if camera_meas
            else np.empty((0, 2))
        )

        all_ais = (
            np.array(
                [
                    [
                        m["north_m"],
                        m["east_m"],
                    ]
                    for m in ais_meas
                ],
                dtype=float,
            )
            if ais_meas
            else np.empty((0, 2))
        )

        active_tids = []

        tracks_pred_meas_radar = []
        tracks_pred_meas_camera = []
        tracks_pred_meas_ais = []

        S_radar = []
        S_camera = []
        S_ais = []

        gated_per_track = {
            "radar": {},
            "camera": {},
            "ais": {},
        }

        H_ais = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=float,
        )

        for tid in list(self.trackers.keys()):
            if t <= self.prev_times[tid]:
                continue

            active_tids.append(tid)

            dt = t - self.prev_times[tid]

            self.trackers[tid].predict(dt)
            self.prev_times[tid] = t

            x = self.trackers[tid].x
            P = self.trackers[tid].get_covariance()

            # Radar predicted measurement
            H_r = self.cfm.H(x, "radar")
            z_pred_r = self.cfm.h(x, "radar")
            S_r = H_r @ P @ H_r.T + self.cfm.R("radar")

            tracks_pred_meas_radar.append(z_pred_r)
            S_radar.append(S_r)

            gated_per_track["radar"][tid] = gating(
                all_radar,
                z_pred_r,
                S_r,
                wrap_bearing=True,
            )

            # Camera predicted measurement
            H_c = self.cfm.H(x, "camera")
            z_pred_c = self.cfm.h(x, "camera")
            S_c = H_c @ P @ H_c.T + self.cfm.R("camera")

            tracks_pred_meas_camera.append(z_pred_c)
            S_camera.append(S_c)

            gated_per_track["camera"][tid] = gating(
                all_camera,
                z_pred_c,
                S_c,
                wrap_bearing=True,
            )

            # AIS predicted measurement: Cartesian N/E position
            z_pred_a = x[:2]
            S_a = H_ais @ P @ H_ais.T + self.cfm.R("ais")

            tracks_pred_meas_ais.append(z_pred_a)
            S_ais.append(S_a)

            gated_per_track["ais"][tid] = gating(
                all_ais,
                z_pred_a,
                S_a,
                wrap_bearing=False,
            )

        assignments, unassigned_tracks, unassigned_meas, C = gnn_hungarian(
            tracks_pred_meas_radar,
            tracks_pred_meas_camera,
            tracks_pred_meas_ais,
            all_radar,
            all_camera,
            all_ais,
            S_radar,
            S_camera,
            S_ais,
        )

        for track_idx, _global_idx, _cost, sensor_id, local_idx in assignments:
            tid = active_tids[track_idx]

            if sensor_id == "radar":
                meas_array = all_radar
                meas_dict = radar_meas[local_idx]

            elif sensor_id == "camera":
                meas_array = all_camera
                meas_dict = camera_meas[local_idx]

            else:
                meas_array = all_ais
                meas_dict = ais_meas[local_idx]

            if sensor_id == "ais":
                self.trackers[tid].update_ned(meas_array[local_idx])
            else:
                self.trackers[tid].update(
                    meas_array[local_idx],
                    sensor_id=sensor_id,
                )

            if (
                not meas_dict.get("is_false_alarm", False)
                and "target_id" in meas_dict
                and tid not in self._gt_label
            ):
                self._gt_label[tid] = meas_dict["target_id"]

        return {
            "assignments": assignments,
            "unassigned_tracks": unassigned_tracks,
            "unassigned_meas": unassigned_meas,
            "active_tids": active_tids,
            "all_radar": all_radar,
            "all_camera": all_camera,
            "all_ais": all_ais,
            "gated_per_track": gated_per_track,
            "cost_matrix": C,
        }


def print_scan_summary(
    t,
    assignments,
    unassigned_tracks,
    unassigned_meas,
    active_tids,
    all_radar,
    all_camera,
):
    print(
        f"\nt={t:.1f}s  assignments: "
        f"{[(a[0], a[3], a[4]) for a in assignments]}"
    )

    for track_idx in unassigned_tracks:
        print(
            f"t={t:.1f}s  MISSED DETECTION: "
            f"track {active_tids[track_idx]}"
        )

    n_radar = len(all_radar)
    n_camera = len(all_camera) if all_camera is not None else 0
    for meas_idx in unassigned_meas:
        if meas_idx < n_radar:
            print(f"t={t:.1f}s  UNMATCHED radar idx={meas_idx}")

        elif meas_idx < n_radar + n_camera:
            print(f"t={t:.1f}s  UNMATCHED camera idx={meas_idx - n_radar}")

        else:
            print(
                f"t={t:.1f}s  UNMATCHED ais idx="
                f"{meas_idx - n_radar - n_camera}"
            )
        


def check_gating_sensor(
    t,
    sensor_id,
    all_meas,
    meas_list,
    active_tids,
    gated_per_track,
    mtt,
):
    n_detected = 0
    n_in_gate = 0

    for tid in active_tids:
        gated_meas = gated_per_track[sensor_id].get(tid, [])

        if tid not in mtt._gt_label:
            continue

        expected_gt = mtt._gt_label[tid]

        n_true = sum(
            1
            for z, m in zip(all_meas, meas_list)
            if any(np.allclose(z, gz) for gz in gated_meas)
            and not m.get("is_false_alarm", False)
            and m.get("target_id") == expected_gt
        )

        n_fa = sum(
            1
            for z, m in zip(all_meas, meas_list)
            if any(np.allclose(z, gz) for gz in gated_meas)
            and m.get("is_false_alarm", False)
        )

        print(
            f"t={t:6.1f}s  [{sensor_id}] track {tid} / GT {expected_gt}  |  "
            f"gate size={len(gated_meas)}  true_det={n_true}  FA_leaked={n_fa}"
        )

        assert n_true <= 1, (
            f"t={t} [{sensor_id}] track {tid}: "
            f"{n_true} true detections in gate, expected <= 1"
        )

        target_detected = any(
            not m.get("is_false_alarm", False)
            and m.get("target_id") == expected_gt
            for m in meas_list
        )

        if (t < 40.0 or 48.0 <= t <= 72.0) and target_detected:
            n_detected += 1

            if n_true == 1:
                n_in_gate += 1

    return n_detected, n_in_gate


def check_association(
    t,
    assignments,
    unassigned_tracks,
    active_tids,
    radar_meas,
    camera_meas,
    mtt,
):
    identity_swaps = 0

    for track_idx, _, _, sensor_id, local_idx in assignments:
        tid = active_tids[track_idx]

        if sensor_id == "radar":
            m = radar_meas[local_idx]

        elif sensor_id == "camera":
            m = camera_meas[local_idx]

        else:
            continue

        if m.get("is_false_alarm", False):
            continue

        if tid not in mtt._gt_label:
            continue

        expected_gt = mtt._gt_label[tid]
        actual_gt = m.get("target_id")

        if actual_gt != expected_gt and 48.0 <= t <= 72.0:
            print(
                f"WARNING t={t}: track {tid} expected GT {expected_gt}, "
                f"got GT {actual_gt} -- identity swap at crossing"
            )

            identity_swaps += 1

    for track_idx in unassigned_tracks:
        tid = active_tids[track_idx]

        if tid not in mtt._gt_label:
            continue

        expected_gt = mtt._gt_label[tid]

        true_exists = any(
            not m.get("is_false_alarm", False)
            and m.get("target_id") == expected_gt
            for m in radar_meas + camera_meas
        )

        if true_exists:
            print(
                f"WARNING t={t}: track {tid} / GT {expected_gt} "
                f"unassigned but measurement exists"
            )

    return identity_swaps