import json
from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from T6_gating_data_association import (
    MultiTargetTracker,
    check_association,
    check_gating_sensor,
    ospa,
    print_scan_summary,
)
from T2_CoordinateFrameManager import CoordinateFrameManager

# set True for simulation, False for real data
SIMULATION_MODE = True

# CONFIRMATION_M = 5 # FOR REAL SCENARIO

CONFIRMATION_M = 3 if SIMULATION_MODE else 10  # 3 for simulation, 5 for real scenario
CONFIRMATION_N = 5 if SIMULATION_MODE else 12  # 5 for simulation, 10 for real scenario

M_DELETE_CONFIRMED = 6
M_DELETE_TENTATIVE = 4
M_DELETE_EKF_TENT  = 4
PRE_GATE_M         = 50.0   # = 25 FOR REAL SCENARIO,50 FOR SIMULATION
MAX_INIT_SPEED_MS  = 5.0   # = 5  FOR REAL SCENARIO (harbour ~10 kn), 25 FOR SIMUALTION 
OSPA_GRACE_SCANS   = 10     # scans to skip before checking per-scan OSPA limit
MOTP_MAX_DIST      = 100.0  # distance threshold: pairs beyond this don't count in MOTP

CONFIRMED_MERGE_POS_M = 12.0
CONFIRMED_MERGE_VEL_MS = 1.5
CONFIRMED_PERSISTENT_MERGE_POS_M = 18.0
CONFIRMED_PERSISTENT_MERGE_SCANS = 3
CONFIRMED_INIT_SUPPRESS_M = 18.0
EKF_INIT_SUPPRESS_M = 12.0
TRACK_POSITION_HISTORY_N = 30
NON_AIS_CONFIRM_MIN_DISPLACEMENT_M = 15.0
NON_AIS_CONFIRM_MIN_HITS = 14
NON_AIS_CONFIRM_MIN_AGE_S = 45.0
STATIC_CLUTTER_MIN_HITS = 12
STATIC_CLUTTER_MAX_DISPLACEMENT_M = 15.0
STATIC_CLUTTER_SUPPRESS_M = 25.0
FRAGMENT_MAX_HITS = 16
FRAGMENT_MAX_AGE_S = 45.0
FRAGMENT_NEAR_STRONG_TRACK_M = 60.0
STRONG_TRACK_MIN_HITS = 25
CE_GRACE_TIME_S = 10.0

class TentativeTrack:
    """Pre-EKF buffer: collects NED positions until velocity can be estimated."""

    def __init__(self, ned_pos, timestamp, sensor_R_cart, sensor_id):
        self.detections = [np.asarray(ned_pos, dtype=float)]
        self.timestamps = [float(timestamp)]
        self.sensor_ids = [sensor_id]
        self.sensor_R = sensor_R_cart
        self.missed = 0

    def add_detection(self, ned_pos, timestamp, sensor_id):
        self.detections.append(np.asarray(ned_pos, dtype=float))
        self.timestamps.append(float(timestamp))
        self.sensor_ids.append(sensor_id)
        self.missed = 0

    def ready_to_initialize(self):
        if len(self.detections) < 2 or self.timestamps[-1] <= self.timestamps[0]:
            return False
        dt    = self.timestamps[-1] - self.timestamps[0]
        speed = float(np.linalg.norm(self.detections[-1] - self.detections[0]) / dt)
        return speed < MAX_INIT_SPEED_MS

    def initialize_state_and_covariance(self):
        z1, z2 = self.detections[0], self.detections[-1]
        t1, t2 = self.timestamps[0], self.timestamps[-1]
        dt = t2 - t1

        if dt <= 0:
            raise ValueError("dt must be positive when initializing velocity.")

        x0 = np.array(
            [
                z2[0],
                z2[1],
                (z2[0] - z1[0]) / dt,
                (z2[1] - z1[1]) / dt,
            ],
            dtype=float,
        )

        P0 = np.zeros((4, 4))
        P0[0:2, 0:2] = self.sensor_R[0:2, 0:2]
        P0[2, 2] = 2.0 * self.sensor_R[0, 0] / dt**2
        P0[3, 3] = 2.0 * self.sensor_R[1, 1] / dt**2

        return x0, P0


class TrackManager(MultiTargetTracker):
    """
    Full track lifecycle:
      pre-EKF TentativeTrack -> tentative EKF -> confirmed -> deleted

    Confirmation uses M-of-N logic:
      confirmed if at least CONFIRMATION_M hits occur within CONFIRMATION_N scans.
    """

    def __init__(self, cfm: CoordinateFrameManager):
        super().__init__(cfm)

        self.tentative_tracks = []
        self._hits = {}
        self._misses = {}

        self._history = {}
        self._confirmed = set()
        self._duplicate_pair_hits = {}
        self._sensor_hits = {}
        self._position_history = {}
        self._track_start_times = {}
        self._static_clutter_points = []

    def _is_confirmed(self, tid):
        if tid in self._confirmed:
            return True

        history = self._history.get(tid)

        if (
            history is not None
            and sum(history) >= CONFIRMATION_M
            and self._has_confirmation_motion_or_ais(tid)
        ):
            self._confirmed.add(tid)
            return True

        return False

    @property
    def confirmed_tracks(self):
        return {
            tid: trk
            for tid, trk in self.trackers.items()
            if self._is_confirmed(tid)
        }

    def _delete_track(self, tid):
        del self.trackers[tid]
        del self.prev_times[tid]

        self._hits.pop(tid, None)
        self._misses.pop(tid, None)
        self._history.pop(tid, None)
        self._confirmed.discard(tid)
        self._gt_label.pop(tid, None)
        self._sensor_hits.pop(tid, None)
        self._position_history.pop(tid, None)
        self._track_start_times.pop(tid, None)
        self._duplicate_pair_hits = {
            pair: count
            for pair, count in self._duplicate_pair_hits.items()
            if tid not in pair
        }

    def _delete_tracks_with_excessive_misses(self, t):
        for tid in list(self.trackers.keys()):
            if self._is_confirmed(tid):
                threshold = M_DELETE_CONFIRMED
            else:
                threshold = M_DELETE_EKF_TENT

            if self._misses.get(tid, 0) >= threshold:
                print(f"t={t:.1f}s  DELETE stale track {tid}")
                self._delete_track(tid)

    def _measurement_inside_any_assigned_gate(
        self,
        z,
        sensor_id,
        gated_per_track,
        assigned_tids,
    ):
        for tid in assigned_tids:
            gated = gated_per_track[sensor_id].get(tid, [])

            if any(np.allclose(z, gz) for gz in gated):
                return True

        return False

    def _track_strength(self, tid):
        P = self.trackers[tid].get_covariance()

        return (
            self._hits.get(tid, 0),
            -self._misses.get(tid, 0),
            -float(np.trace(P)),
        )

    def _duplicate_pair_key(self, tid_i, tid_j):
        return tuple(sorted((tid_i, tid_j)))

    def _track_near_position(self, ned_pos, max_dist, confirmed_only=False):
        if confirmed_only:
            tracks = self.confirmed_tracks.values()
        else:
            tracks = self.trackers.values()

        for trk in tracks:
            dist = float(np.linalg.norm(ned_pos - trk.get_state()[:2]))

            if dist < max_dist:
                return True

        return False

    def _sensor_hit_counts_from_ids(self, sensor_ids):
        counts = {}

        for sensor_id in sensor_ids:
            counts[sensor_id] = counts.get(sensor_id, 0) + 1

        return counts

    def _has_ais_support(self, tid):
        return self._sensor_hits.get(tid, {}).get("ais", 0) > 0

    def _record_track_position(self, tid):
        self._position_history.setdefault(
            tid,
            deque(maxlen=TRACK_POSITION_HISTORY_N),
        ).append(self.trackers[tid].get_state()[:2].copy())

    def _track_displacement(self, tid):
        history = self._position_history.get(tid)

        if history is None or len(history) < 2:
            return 0.0

        return float(np.linalg.norm(history[-1] - history[0]))

    def _track_age(self, tid):
        if tid not in self._track_start_times:
            return 0.0

        return float(self.prev_times[tid] - self._track_start_times[tid])

    def _has_confirmation_motion_or_ais(self, tid):
        if SIMULATION_MODE:
            return True

        if self._has_ais_support(tid):
            return True

        return (
            self._hits.get(tid, 0) >= NON_AIS_CONFIRM_MIN_HITS
            and self._track_age(tid) >= NON_AIS_CONFIRM_MIN_AGE_S
            and self._track_displacement(tid) >= NON_AIS_CONFIRM_MIN_DISPLACEMENT_M
        )

    def _near_stronger_track(self, tid):
        x = self.trackers[tid].get_state()

        for other_tid, other_trk in self.trackers.items():
            if other_tid == tid:
                continue

            if self._hits.get(other_tid, 0) < STRONG_TRACK_MIN_HITS:
                continue

            if self._hits.get(other_tid, 0) <= self._hits.get(tid, 0):
                continue

            dist = float(np.linalg.norm(x[:2] - other_trk.get_state()[:2]))

            if dist < FRAGMENT_NEAR_STRONG_TRACK_M:
                return True

        return False

    def _near_static_clutter(self, ned_pos):
        return any(
            float(np.linalg.norm(ned_pos - clutter_pos))
            < STATIC_CLUTTER_SUPPRESS_M
            for clutter_pos in self._static_clutter_points
        )

    def _register_static_clutter(self, ned_pos):
        if self._near_static_clutter(ned_pos):
            return

        self._static_clutter_points.append(np.asarray(ned_pos, dtype=float).copy())

    def _delete_static_clutter_tracks(self, t):
        for tid in list(self.trackers.keys()):
            if self._has_ais_support(tid):
                continue

            if self._hits.get(tid, 0) < STATIC_CLUTTER_MIN_HITS:
                continue

            if self._track_displacement(tid) >= STATIC_CLUTTER_MAX_DISPLACEMENT_M:
                continue

            self._register_static_clutter(self.trackers[tid].get_state()[:2])
            print(f"t={t:.1f}s  DELETE stationary clutter track {tid}")
            self._delete_track(tid)

    def _delete_short_fragment_tracks(self, t):
        for tid in list(self.trackers.keys()):
            if self._has_ais_support(tid):
                continue

            if self._hits.get(tid, 0) > FRAGMENT_MAX_HITS:
                continue

            if self._track_age(tid) > FRAGMENT_MAX_AGE_S:
                continue

            if not self._near_stronger_track(tid):
                continue

            print(f"t={t:.1f}s  DELETE short fragment track {tid}")
            self._delete_track(tid)

    def _merge_duplicates(self):
        gate = chi2.ppf(0.99, df=4)
        confirmed_gate = chi2.ppf(0.95, df=4)
        tids = list(self.trackers.keys())
        to_delete = set()

        for i in range(len(tids)):
            tid_i = tids[i]

            if tid_i in to_delete:
                continue

            x_i = self.trackers[tid_i].get_state()
            P_i = self.trackers[tid_i].get_covariance()
            conf_i = self._is_confirmed(tid_i)

            for j in range(i + 1, len(tids)):
                tid_j = tids[j]

                if tid_j in to_delete:
                    continue

                conf_j = self._is_confirmed(tid_j)

                x_j = self.trackers[tid_j].get_state()
                P_j = self.trackers[tid_j].get_covariance()

                diff = x_i - x_j

                try:
                    d2 = float(diff @ np.linalg.solve(P_i + P_j, diff))
                except np.linalg.LinAlgError:
                    continue

                pos_dist = float(np.linalg.norm(diff[:2]))
                vel_dist = float(np.linalg.norm(diff[2:]))

                if conf_i and conf_j:
                    pair_key = self._duplicate_pair_key(tid_i, tid_j)

                    immediate_duplicate = (
                        d2 < confirmed_gate
                        and pos_dist < CONFIRMED_MERGE_POS_M
                        and vel_dist < CONFIRMED_MERGE_VEL_MS
                    )

                    persistent_duplicate_candidate = (
                        pos_dist < CONFIRMED_PERSISTENT_MERGE_POS_M
                        and (
                            d2 < gate
                            or pos_dist < CONFIRMED_MERGE_POS_M
                        )
                    )

                    if persistent_duplicate_candidate:
                        self._duplicate_pair_hits[pair_key] = (
                            self._duplicate_pair_hits.get(pair_key, 0) + 1
                        )
                    else:
                        self._duplicate_pair_hits.pop(pair_key, None)

                    persistent_duplicate = (
                        self._duplicate_pair_hits.get(pair_key, 0)
                        >= CONFIRMED_PERSISTENT_MERGE_SCANS
                    )

                    if not immediate_duplicate and not persistent_duplicate:
                        continue

                elif d2 >= gate:
                    continue

                hits_i = self._hits.get(tid_i, 0)
                hits_j = self._hits.get(tid_j, 0)

                if conf_i and conf_j:
                    if self._track_strength(tid_i) >= self._track_strength(tid_j):
                        to_delete.add(tid_j)
                    else:
                        to_delete.add(tid_i)
                        break

                elif conf_i and not conf_j:
                    to_delete.add(tid_j)

                elif conf_j and not conf_i:
                    to_delete.add(tid_i)
                    break

                elif hits_i >= hits_j:
                    to_delete.add(tid_j)

                else:
                    to_delete.add(tid_i)
                    break

        for tid in to_delete:
            print(f"MERGE  delete duplicate track {tid}")
            self._delete_track(tid)

    def _detection_to_ned_and_covariance(self, detection, sensor_id):
        if sensor_id == "ais":
            ned = np.array(
                [
                    float(detection[0]),
                    float(detection[1]),
                ],
                dtype=float,
            )
            R_cart = self.cfm.R("ais")

        else:
            ned = self.cfm.polar_to_ned(
                float(detection[0]),
                float(detection[1]),
                sensor_id,
            )
            R_cart = self.cfm.R_cartesian(ned, sensor_id)

        return ned, R_cart

    def process_unassigned_detection(self, detection, timestamp, sensor_id):
        """Match detection to an existing TentativeTrack or start a new one."""

        ned, R_cart = self._detection_to_ned_and_covariance(
            detection,
            sensor_id,
        )

        best_pt, best_dist = None, PRE_GATE_M

        for pt in self.tentative_tracks:
            dist = float(np.linalg.norm(ned - pt.detections[-1]))

            if dist < best_dist:
                best_dist, best_pt = dist, pt

        if best_pt is not None:
            best_pt.add_detection(ned, timestamp, sensor_id)

            if best_pt.ready_to_initialize():
                x0, P0 = best_pt.initialize_state_and_covariance()
                tid = self._init_track(x0, P0, timestamp)

                self._hits[tid] = 2
                self._misses[tid] = 0
                self._history[tid] = deque([1, 1], maxlen=CONFIRMATION_N)
                self._sensor_hits[tid] = self._sensor_hit_counts_from_ids(
                    best_pt.sensor_ids
                )
                self._position_history[tid] = deque(
                    [p.copy() for p in best_pt.detections],
                    maxlen=TRACK_POSITION_HISTORY_N,
                )
                self._track_start_times[tid] = best_pt.timestamps[0]

                self.tentative_tracks.remove(best_pt)

        else:
            self.tentative_tracks.append(
                TentativeTrack(ned, timestamp, R_cart, sensor_id)
            )

            print(f"t={timestamp:.1f}s  INIT  pre_track from {sensor_id}")

    def step(self, t, radar_meas, camera_meas, gnss_meas=None, ais_meas=None):
        if ais_meas is None:
            ais_meas = []

        for pt in self.tentative_tracks:
            pt.missed += 1

        result = super().step(t, radar_meas, camera_meas, gnss_meas, ais_meas)

        assigned_tids = {
            result["active_tids"][track_idx]
            for track_idx, *_ in result["assignments"]
        }

        for track_idx, _global_idx, _cost, sensor_id, _local_idx in result["assignments"]:
            tid = result["active_tids"][track_idx]

            self._hits[tid] = self._hits.get(tid, 0) + 1
            self._misses[tid] = 0
            self._sensor_hits.setdefault(tid, {})
            self._sensor_hits[tid][sensor_id] = (
                self._sensor_hits[tid].get(sensor_id, 0) + 1
            )
            self._record_track_position(tid)

            self._history.setdefault(
                tid,
                deque(maxlen=CONFIRMATION_N),
            ).append(1)

        any_sensor_active = (
            len(result["all_radar"]) > 0
            or len(result["all_camera"]) > 0
            or len(ais_meas) > 0
        )

        if any_sensor_active:
            for track_idx in result["unassigned_tracks"]:
                tid = result["active_tids"][track_idx]

                self._misses[tid] = self._misses.get(tid, 0) + 1

                self._history.setdefault(
                    tid,
                    deque(maxlen=CONFIRMATION_N),
                ).append(0)

        n_radar = len(result["all_radar"])
        n_camera = len(result["all_camera"])
        n_ais = len(result["all_ais"])

        for meas_idx in result["unassigned_meas"]:

            if meas_idx < n_radar:
                sensor_id = "radar"
                local_idx = meas_idx
                det_dict = radar_meas[local_idx]
                z = result["all_radar"][local_idx]

                detection = [
                    det_dict["range_m"],
                    det_dict["bearing_rad"],
                ]

            elif meas_idx < n_radar + n_camera:
                sensor_id = "camera"
                local_idx = meas_idx - n_radar
                det_dict = camera_meas[local_idx]
                z = result["all_camera"][local_idx]

                detection = [
                    det_dict["range_m"],
                    det_dict["bearing_rad"],
                ]

            else:
                sensor_id = "ais"
                local_idx = meas_idx - n_radar - n_camera
                det_dict = ais_meas[local_idx]
                z = result["all_ais"][local_idx]

                detection = [
                    det_dict["north_m"],
                    det_dict["east_m"],
                ]

            inside_assigned_gate = self._measurement_inside_any_assigned_gate(
                z=z,
                sensor_id=sensor_id,
                gated_per_track=result["gated_per_track"],
                assigned_tids=assigned_tids,
            )

            if inside_assigned_gate:
                print(
                    f"t={t:.1f}s  suppress duplicate initiation from "
                    f"{sensor_id} idx={local_idx}"
                )
                continue

            if sensor_id != "ais":
                ned, _ = self._detection_to_ned_and_covariance(
                    detection,
                    sensor_id,
                )

                if self._near_static_clutter(ned):
                    print(
                        f"t={t:.1f}s  suppress static-clutter initiation from "
                        f"{sensor_id} idx={local_idx}"
                    )
                    continue

                if self._track_near_position(
                    ned,
                    CONFIRMED_INIT_SUPPRESS_M,
                    confirmed_only=True,
                ):
                    print(
                        f"t={t:.1f}s  suppress near-confirmed initiation from "
                        f"{sensor_id} idx={local_idx}"
                    )
                    continue

                if self._track_near_position(
                    ned,
                    EKF_INIT_SUPPRESS_M,
                    confirmed_only=False,
                ):
                    print(
                        f"t={t:.1f}s  suppress near-EKF initiation from "
                        f"{sensor_id} idx={local_idx}"
                    )
                    continue

            self.process_unassigned_detection(
                detection=detection,
                timestamp=t,
                sensor_id=sensor_id,
            )



        self.tentative_tracks = [
            pt
            for pt in self.tentative_tracks
            if pt.missed < M_DELETE_TENTATIVE
        ]

        # self._delete_static_clutter_tracks(t)
        # self._delete_short_fragment_tracks(t)
        self._delete_tracks_with_excessive_misses(t)
        self._merge_duplicates()

        print(
            f"t={t:.1f}s  EKF hits={dict(self._hits)}  "
            f"pre_tracks={len(self.tentative_tracks)}"
        )

        return result


    def assert_end_of_run(
        self,
        target_ids,
        data,
        ospa_crossing,
        ospa_post,
        motp_distances,
        motp_matches,
        identity_swap,
        ce_series,
        motp_series,
    ):
        confirmed = self.confirmed_tracks
        n_confirmed = len(confirmed)
        n_gt = len(target_ids)
        ce_final = abs(n_confirmed - n_gt)

        mean_ce = float(np.mean(ce_series)) if ce_series else float("nan")

        print(f"\nCE time series (per scan): {[int(v) for v in ce_series]}")
        print(f"Mean CE (full run): {mean_ce:.3f}")
        print(f"Final CE: {ce_final}  (confirmed={n_confirmed}, expected={n_gt})")

        assert mean_ce <= 0.5, (
            f"Mean CE {mean_ce:.3f} > 1.0 -- poor cardinality throughout run"
        )

        assert ce_final <= 1, f"Cardinality error {ce_final} > 1"

        if confirmed:
            est_final = np.array(
                [trk.get_state()[:2] for trk in confirmed.values()]
            )

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

            print("\nFinal position RMSE (confirmed track -> nearest GT target):")

            for ri, ci in zip(r_ind, c_ind):
                rmse = D[ri, ci]
                print(
                    f"  confirmed track -> GT {target_ids[ci]}: "
                    f"RMSE = {rmse:.2f} m"
                )

                assert rmse < 20.0, (
                    f"Final RMSE {rmse:.2f} m > 20 m -- track lost"
                )

        if ospa_crossing:
            mean_ospa_crossing = float(np.mean(ospa_crossing))
            print(
                f"\nMean OSPA during crossing (48-72s): "
                f"{mean_ospa_crossing:.2f} m"
            )

            assert mean_ospa_crossing < 40.0, (
                f"Mean OSPA crossing {mean_ospa_crossing:.2f} m > 40 m"
            )

        if ospa_post:
            mean_ospa_post = float(np.mean(ospa_post))
            print(f"Mean OSPA after crossing (>72s):   {mean_ospa_post:.2f} m")

            assert mean_ospa_post < 40.0, (
                f"Mean OSPA post {mean_ospa_post:.2f} m > 40 m"
            )

        if motp_series:
            print(
                f"\nMOTP time series (per scan, m): "
                f"{[f'{v:.1f}' for v in motp_series]}"
            )

        if motp_matches > 0:
            motp_scalar = float(np.sum(motp_distances) / motp_matches)
            print(f"Mean MOTP (full run): {motp_scalar:.2f} m")

            assert motp_scalar < 15.0, (
                f"MOTP {motp_scalar:.2f} m > 40 m -- localisation too poor"
            )

        if identity_swap > 0:
            print(f"\nIdentity swaps detected: {identity_swap}")
            assert False, (
                f"{identity_swap} identity swaps detected -- check associations"
            )


def test_gating_scenario(
    json_path: str,
    ospa_limit: float = 50.0,
) -> None:
    with open(json_path) as f:
        data = json.load(f)

    cfm = CoordinateFrameManager(
        camera_offset=np.array([-80.0, 120.0]),
        radar_R=np.diag([5.0**2, np.deg2rad(0.3) ** 2]),
        camera_R=np.diag([8.0**2, np.deg2rad(0.15) ** 2]),
        ais_R=np.diag([10.0**2, 10.0**2]),
    )

    cfm.update_vessel_position(np.array([0.0, 0.0]))

    target_ids = [int(k) for k in data["ground_truth"].keys()]
    tm = TrackManager(cfm)

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
            if m["sensor_id"] in ("radar", "camera", "gnss", "ais")
        )
    )

    ospa_series = []
    ce_series = []
    motp_series = []
    motp_distances = []
    motp_matches = 0
    identity_swap = 0
    n_detected_gate = 0
    n_in_gate = 0

    for t in scan_times:
        radar_meas = [
            m for m in data["measurements"]
            if m["sensor_id"] == "radar" and m["time"] == t
        ]

        camera_meas = [
            m for m in data["measurements"]
            if m["sensor_id"] == "camera" and m["time"] == t
        ]

        gnss_meas = [
            m for m in data["measurements"]
            if m["sensor_id"] == "gnss" and m["time"] == t
        ]

        ais_meas = [
            m for m in data["measurements"]
            if m["sensor_id"] == "ais" and m["time"] == t
        ]

        result = tm.step(t, radar_meas, camera_meas, gnss_meas, ais_meas)

        print_scan_summary(
            t,
            result["assignments"],
            result["unassigned_tracks"],
            result["unassigned_meas"],
            result["active_tids"],
            result["all_radar"],
            result["all_camera"],

        )

        identity_swap += check_association(
            t,
            result["assignments"],
            result["unassigned_tracks"],
            result["active_tids"],
            radar_meas,
            camera_meas,
            tm,
        )

        nd, ng = check_gating_sensor(
            t,
            "radar",
            result["all_radar"],
            radar_meas,
            result["active_tids"],
            result["gated_per_track"],
            tm,
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
            tm,
        )
        n_detected_gate += nd
        n_in_gate += ng

        confirmed = tm.confirmed_tracks

        sensor_active = (
            len(radar_meas) > 0
            or len(camera_meas) > 0
            or len(ais_meas) > 0
        )

        if sensor_active and t >= CE_GRACE_TIME_S:
            # target active if t is within its GT time span
            active_target_ids = [
                tid
                for tid in target_ids
                if gt_lookup[tid][0][0] <= t <= gt_lookup[tid][-1][0]
            ]
            if not active_target_ids:
                active_target_ids = target_ids

            ce_t = abs(len(confirmed) - len(active_target_ids))
            ce_series.append(ce_t)

            if confirmed:
                est_pos = [trk.get_state()[:2] for trk in confirmed.values()]
                gt_pos  = [gt_pos_at(tid, t) for tid in active_target_ids]

                ospa_t = ospa(est_pos, gt_pos, c=100.0, p=2)
                ospa_series.append(ospa_t)

                D = np.array([[np.linalg.norm(e - g) for g in gt_pos] for e in est_pos])
                r_ind, c_ind = linear_sum_assignment(D)

                # MOTP: only count pairs within threshold (excludes ghost/FA tracks)
                scan_dists = [D[ri, ci] for ri, ci in zip(r_ind, c_ind)
                              if D[ri, ci] < MOTP_MAX_DIST]
                if scan_dists:
                    motp_t = float(np.mean(scan_dists))
                    motp_series.append(motp_t)
                    for d in scan_dists:
                        motp_distances.append(d)
                        motp_matches += 1
                else:
                    motp_t = float("nan")

                print(
                    f"t={t:6.1f}s  OSPA={ospa_t:.2f} m  "
                    f"CE={ce_t}  MOTP={motp_t:.1f} m  "
                    f"confirmed={len(confirmed)}  active_gt={len(active_target_ids)}"
                )

    if n_detected_gate > 0:
        print(f"\nEmpirical gate inclusion rate: {n_in_gate / n_detected_gate:.3f}")

    mean_ce = float(np.mean(ce_series)) if ce_series else float("nan")
    mean_ospa = float(np.mean(ospa_series)) if ospa_series else float("nan")
    mean_motp = float(np.mean(motp_distances)) if motp_distances else float("nan")

    # print(f"\nScenario: {json_path}")
    # print(f"CE time series: {[int(v) for v in ce_series]}")
    # print(f"Mean CE: {mean_ce:.3f}")

    # print(f"\nOSPA time series: {[f'{v:.1f}' for v in ospa_series]}")
    # print(f"Mean OSPA: {mean_ospa:.2f} m")

    # print(f"\nMOTP time series: {[f'{v:.1f}' for v in motp_series]}")
    # print(f"Mean MOTP: {mean_motp:.2f} m")

    # assert mean_ce <= 1.0, f"Mean CE {mean_ce:.3f} > 1.0"
    # ospa_steady = ospa_series[OSPA_GRACE_SCANS:]
    # if ospa_steady:
    #     mean_ospa_steady = float(np.mean(ospa_steady))
    #     print(f"Mean OSPA (after grace period): {mean_ospa_steady:.2f} m")
    #     assert mean_ospa_steady < ospa_limit, (
    #         f"Mean OSPA {mean_ospa_steady:.2f} m >= {ospa_limit:.1f} m after grace period"
    #     )

    # if identity_swap > 0:
    #     assert False, f"{identity_swap} identity swaps detected"

    tm.assert_end_of_run(
        target_ids=target_ids,
        data=data,
        ospa_crossing=ospa_series[48:73],
        ospa_post=ospa_series[73:],
        motp_distances=motp_distances,
        motp_matches=motp_matches,
        identity_swap=identity_swap,
        ce_series=ce_series,
        motp_series=motp_series,
    )
    

if __name__ == "__main__":


    test_gating_scenario(
        "harbour_sim_output/scenario_D.json",
        ospa_limit=50.0,
    )
    
