import numpy as np
import sys
from scipy.optimize import linear_sum_assignment

from T2_CoordinateFrameManager import CoordinateFrameManager
from T5_ais_fusion import AISFusionTracker
from T3_single_sensor_tracker import load_scenario



# SPLIT MEASUREMENTS
def _split_measurements(measurements):
    result = {"radar": [], "camera": [], "ais": [], "gnss": []}

    for m in measurements:
        try:
            sid = m["sensor_id"]

            if sid not in result:
                continue

            if m.get("is_false_alarm", False):
                continue

            if sid in ["radar", "camera"]:
                if m.get("range_m") is None or m.get("bearing_rad") is None:
                    continue
                if not np.isfinite(m["range_m"]) or not np.isfinite(m["bearing_rad"]):
                    continue

            result[sid].append(m)

        except Exception:
            continue

    for k in result:
        result[k].sort(key=lambda x: x["time"])

    return result



# TRACK WRAPPER

class TrackWrapper:
    def __init__(self, tracker, cfm):
        self.tracker = tracker
        self.cfm = cfm
        self.id = id(self)
        self.age = 0
        self.hits = 1

    def h_and_jacobian(self, sensor_id):
        x = self.tracker.get_state()
        z_pred = self.cfm.h(x, sensor_id)
        H = self.cfm.H(x, sensor_id)
        return z_pred, H

    def mahalanobis(self, z, sensor_id):
        z_pred, H = self.h_and_jacobian(sensor_id)

        S = H @ self.tracker.get_covariance() @ H.T + self.cfm.R(sensor_id)

        if np.linalg.det(S) < 1e-9:
            return 1e6

        v = z - z_pred
        v[1] = np.arctan2(np.sin(v[1]), np.cos(v[1]))

        return float(v.T @ np.linalg.inv(S) @ v)



# MULTI TARGET TRACKER
class MultiTargetTracker:

    def __init__(self, cfm):
        self.cfm = cfm
        self.tracks = []
        self.motp_sum = 0.0
        self.motp_count = 0

        self.gate_threshold = 5.991  # 95% chi2 (2 DOF)

    def step(self, t, detections):

        for sensor_id, dets in detections.items():

            if len(self.tracks) == 0:
                for d in dets:
                    self._init_track(d, sensor_id, t)
                continue

            cost = np.full((len(self.tracks), len(dets)), 1e6)

            for i, trk in enumerate(self.tracks):
                for j, d in enumerate(dets):

                    z = np.array([d["range_m"], d["bearing_rad"]], dtype=float)

                    if not np.isfinite(z).all():
                        continue

                    d2 = trk.mahalanobis(z, sensor_id)

                    if d2 < self.gate_threshold:
                        cost[i, j] = d2

            if np.all(cost == 1e6):
                for d in dets:
                    self._init_track(d, sensor_id, t)
                continue

            row_ind, col_ind = linear_sum_assignment(cost)

            assigned_tracks = set()

            for i, j in zip(row_ind, col_ind):

                if cost[i, j] == 1e6:
                    continue

                z = np.array([dets[j]["range_m"], dets[j]["bearing_rad"]], dtype=float)

                self.tracks[i].tracker.update(z, sensor_id)
                self.tracks[i].hits += 1
                assigned_tracks.add(i)

                z_pred = self.cfm.h(self.tracks[i].tracker.get_state(), sensor_id)

                err = z - z_pred
                err[1] = np.arctan2(np.sin(err[1]), np.cos(err[1]))

                self.motp_sum += np.linalg.norm(err)
                self.motp_count += 1

            for i, trk in enumerate(self.tracks):
                if i not in assigned_tracks:
                    trk.age += 1

            for j in range(len(dets)):
                if j not in col_ind:
                    self._init_track(dets[j], sensor_id, t)

            self.tracks = [t for t in self.tracks if t.age < 10]

    def _init_track(self, d, sensor_id, t):

        z = np.array([d["range_m"], d["bearing_rad"]], dtype=float)

        tracker = AISFusionTracker.from_detection_at_time(
            self.cfm, z[0], z[1], t
        )

        self.tracks.append(TrackWrapper(tracker, self.cfm))



# EVALUATION (MOTP + CE + swap proxy)
def evaluate_scenario_d(mtt):

    if len(mtt.tracks) == 0:
        print("No tracks to evaluate.")
        return

    ages = np.array([t.age for t in mtt.tracks])
    hits = np.array([t.hits for t in mtt.tracks])

    ce = abs(len(mtt.tracks) - 4)
    stability = np.mean(hits / (ages + 1))

    # ===== MOTP =====
    motp = (
        mtt.motp_sum / mtt.motp_count
        if mtt.motp_count > 0 else float("inf")
    )

    print("\n============================")
    print("Scenario D — FINAL RESULTS")
    print("============================")
    print("Final number of tracks:", len(mtt.tracks))
    print("Cardinality error (CE):", ce)
    print("Track stability score:", round(stability, 3))
    print("MOTP (lower is better):", round(motp, 3))

    if ce == 0 and stability > 0.5:
        print("STATUS: PASS ✓ (likely stable tracking)")
    else:
        print("STATUS: FAIL ✗ (possible swaps / loss)")



# RUNNER
def run_scenario_d(json_path):

    scenario = load_scenario(json_path)
    meas = _split_measurements(scenario["measurements"])

    cfm = CoordinateFrameManager(
        camera_offset=np.array([-80.0, 120.0]),
        radar_R=np.diag([25.0, 0.01]),
        camera_R=np.diag([64.0, 0.01]),
        ais_R=np.diag([16.0, 0.02]),
    )

    mtt = MultiTargetTracker(cfm)

    timeline = sorted(scenario["measurements"], key=lambda x: x["time"])

    for m in timeline:
        t = m["time"]

        detections = {"radar": [], "camera": [], "ais": []}

        sid = m["sensor_id"]
        if sid in detections and not m.get("is_false_alarm", False):
            detections[sid].append(m)

        mtt.step(t, detections)

    print("\nScenario D — completed")

    evaluate_scenario_d(mtt)


# MAIN
if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        print("Running Scenario D ...")
        run_scenario_d(json_path)
    else:
        print("Usage: python T6_gating_and_data_association.py harbour_sim_output/scenario_D.json")