"""T4: radar + stereo-camera EKF fusion.

Extends the T3 radar-only EKF with:
- sequential fusion: radar update, then camera update
- centralised fusion: one joint EKF update with stacked measurements

Scenario B validation compares RMSE and NIS consistency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from T2_CoordinateFrameManager import CoordinateFrameManager
from T3_single_sensor_tracker import RadarEKFTracker, wrap_angle, load_scenario


class RadarCameraFusionTracker(RadarEKFTracker):
    """Single-target EKF supporting radar/camera fusion."""

    def update_sequential(
        self,
        radar_z: np.ndarray | None = None,
        camera_z: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Apply radar then camera EKF updates in sequence."""
        nis_values: dict[str, float] = {}

        if radar_z is not None:
            nis_values["radar"] = self.update(radar_z, sensor_id="radar")

        if camera_z is not None:
            nis_values["camera"] = self.update(camera_z, sensor_id="camera")

        return nis_values

    def update_centralised(
        self,
        radar_z: np.ndarray | None = None,
        camera_z: np.ndarray | None = None,
    ) -> float | None:
        """Apply a single joint EKF update using all available measurements."""
        measurements: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        jacobians: list[np.ndarray] = []
        covariances: list[np.ndarray] = []

        if radar_z is not None:
            measurements.append(np.asarray(radar_z, dtype=float).reshape(2))
            predictions.append(self.cfm.h(self.x, "radar"))
            jacobians.append(self.cfm.H(self.x, "radar"))
            covariances.append(self.cfm.R("radar"))

        if camera_z is not None:
            measurements.append(np.asarray(camera_z, dtype=float).reshape(2))
            predictions.append(self.cfm.h(self.x, "camera"))
            jacobians.append(self.cfm.H(self.x, "camera"))
            covariances.append(self.cfm.R("camera"))

        if not measurements:
            return None

        z = np.concatenate(measurements)
        h = np.concatenate(predictions)
        H = np.vstack(jacobians)

        R = np.zeros((z.size, z.size), dtype=float)
        offset = 0
        for cov in covariances:
            n = cov.shape[0]
            R[offset:offset + n, offset:offset + n] = cov
            offset += n

        residual = z - h
        for i in range(1, residual.size, 2):
            residual[i] = wrap_angle(residual[i])

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ residual
        self.P = (np.eye(4) - K @ H) @ self.P

        return float(residual.T @ np.linalg.inv(S) @ residual)


# ---------------------------------------------------------------------------
# Scenario B validation
# ---------------------------------------------------------------------------

def _build_cfm(sensor_configs: dict | None = None) -> CoordinateFrameManager:
    radar_cfg = (sensor_configs or {}).get("radar", {})
    camera_cfg = (sensor_configs or {}).get("camera", {})

    radar_sigma_r = float(radar_cfg.get("sigma_r_m", 5.0))
    radar_sigma_phi = np.deg2rad(float(radar_cfg.get("sigma_phi_deg", 0.3)))
    camera_sigma_r = float(camera_cfg.get("sigma_r_m", 8.0))
    camera_sigma_phi = np.deg2rad(float(camera_cfg.get("sigma_phi_deg", 0.15)))
    camera_offset = np.array(
        camera_cfg.get("pos_ned", [-80.0, 120.0]),
        dtype=float,
    )

    cfm = CoordinateFrameManager(
        camera_offset=camera_offset,
        radar_R=np.diag([radar_sigma_r**2, radar_sigma_phi**2]),
        camera_R=np.diag([camera_sigma_r**2, camera_sigma_phi**2]),
        ais_R=np.eye(2),
    )
    cfm.update_vessel_position(np.array([0.0, 0.0], dtype=float))
    return cfm


def _group_measurements_by_time(measurements: list[dict]) -> dict[float, dict[str, np.ndarray]]:
    """Return true radar/camera measurements grouped by their real timestamp."""
    grouped: dict[float, dict[str, np.ndarray]] = {}

    for m in measurements:
        sensor_id = m["sensor_id"]
        if sensor_id not in ("radar", "camera") or m["is_false_alarm"]:
            continue

        grouped.setdefault(float(m["time"]), {})[sensor_id] = np.array(
            [m["range_m"], m["bearing_rad"]],
            dtype=float,
        )

    return grouped


def _nis_fraction_in_bounds(values: list[float], dof: int) -> float:
    """Fraction of NIS values inside the 95% chi-square consistency bounds."""
    if not values:
        return float("nan")

    # Precomputed chi-square 2.5% and 97.5% quantiles for the dimensions used here.
    bounds = {
        2: (0.0506, 5.991),
        4: (0.711, 9.488),
    }
    lo, hi = bounds[dof]
    arr = np.array(values)
    return float(np.mean((arr >= lo) & (arr <= hi)))


def run_scenario_b(json_path: str | Path, warmup_scans: int = 0) -> dict:
    """
    Run both fusion architectures on Scenario B.

    The radar and camera are asynchronous, so validation is event-driven:
    the EKF predicts to each real sensor timestamp and updates with the
    measurement(s) available at exactly that time. A centralised 4D update is
    only performed when radar and camera are simultaneous.
    """
    scenario = load_scenario(json_path)

    gt_records = scenario["ground_truth"]["0"]
    gt_lookup = {rec[0]: np.array(rec[1:], dtype=float) for rec in gt_records}
    gt_times = np.array(sorted(gt_lookup.keys()))

    measurements_by_time = _group_measurements_by_time(scenario["measurements"])

    results = {}

    for architecture in ("sequential", "centralised"):
        cfm = _build_cfm(scenario.get("sensor_configs"))
        tracker: RadarCameraFusionTracker | None = None
        prev_time: float | None = None

        rmse_sq: list[float] = []
        nis_radar: list[float] = []
        nis_camera: list[float] = []
        nis_centralised_2d: list[float] = []
        nis_centralised_4d: list[float] = []
        scans_with_camera = 0
        update_count = 0

        for t in sorted(measurements_by_time):
            measurements = measurements_by_time[t]
            z_r = measurements.get("radar")
            z_c = measurements.get("camera")

            if tracker is None:
                z0 = z_r if z_r is not None else z_c
                if z0 is None:
                    continue
                tracker = RadarCameraFusionTracker.from_detection(
                    cfm, range_m=z0[0], bearing_rad=z0[1]
                )
                prev_time = t
                continue

            if prev_time is None:
                prev_time = t
                continue

            dt = t - prev_time
            prev_time = t

            tracker.predict(dt)
            update_count += 1

            if update_count <= warmup_scans:
                continue

            if architecture == "sequential":
                nis_dict = tracker.update_sequential(z_r, z_c)

                if "radar" in nis_dict:
                    nis_radar.append(nis_dict["radar"])
                if "camera" in nis_dict:
                    nis_camera.append(nis_dict["camera"])
                    scans_with_camera += 1

            else:
                nis = tracker.update_centralised(z_r, z_c)
                if nis is not None:
                    if z_r is not None and z_c is not None:
                        nis_centralised_4d.append(nis)
                    else:
                        nis_centralised_2d.append(nis)
                if z_c is not None:
                    scans_with_camera += 1

            closest_t = gt_times[np.argmin(np.abs(gt_times - t))]
            gt_state = gt_lookup[closest_t]
            err = np.linalg.norm(tracker.get_state()[:2] - gt_state[:2])
            rmse_sq.append(err**2)

        rmse = float(np.sqrt(np.mean(rmse_sq))) if rmse_sq else float("nan")

        if architecture == "sequential":
            results["sequential"] = {
                "rmse_m": rmse,
                "nis_frac_radar": _nis_fraction_in_bounds(nis_radar, dof=2),
                "nis_frac_camera": _nis_fraction_in_bounds(nis_camera, dof=2),
                "scans_with_camera": scans_with_camera,
            }
        else:
            nis_frac_2d = _nis_fraction_in_bounds(nis_centralised_2d, dof=2)
            nis_frac_4d = _nis_fraction_in_bounds(nis_centralised_4d, dof=4)
            consistent_count = 0.0
            total_nis_count = 0
            if nis_centralised_2d:
                consistent_count += nis_frac_2d * len(nis_centralised_2d)
                total_nis_count += len(nis_centralised_2d)
            if nis_centralised_4d:
                consistent_count += nis_frac_4d * len(nis_centralised_4d)
                total_nis_count += len(nis_centralised_4d)
            centralised_consistent = (
                consistent_count / total_nis_count
                if total_nis_count
                else float("nan")
            )

            results["centralised"] = {
                "rmse_m": rmse,
                "nis_frac_all": centralised_consistent,
                "nis_frac_single_sensor": nis_frac_2d,
                "nis_frac_joint": nis_frac_4d,
                "joint_updates": len(nis_centralised_4d),
                "scans_with_camera": scans_with_camera,
            }

    return results

def print_scenario_b_results(results: dict) -> None:
    seq = results["sequential"]
    cen = results["centralised"]

    print("=" * 60)
    print("Scenario B — Radar + Camera Fusion Comparison (T4)")
    print("=" * 60)
    print(f"{'Metric':<38} {'Sequential':>10} {'Centralised':>10}")
    print("-" * 60)
    print(f"{'Position RMSE (m)':<38} {seq['rmse_m']:>10.2f} {cen['rmse_m']:>10.2f}")
    print(f"{'NIS in bounds — radar chi2(2) (%)':<38} {seq['nis_frac_radar']*100:>10.1f} {'—':>10}")
    print(f"{'NIS in bounds — camera chi2(2) (%)':<38} {seq['nis_frac_camera']*100:>10.1f} {'—':>10}")
    print(f"{'NIS in bounds — central all (%)':<38} {'—':>10} {cen['nis_frac_all']*100:>10.1f}")
    print(f"{'NIS in bounds — central chi2(2) (%)':<38} {'—':>10} {cen['nis_frac_single_sensor']*100:>10.1f}")
    print(f"{'NIS in bounds — joint chi2(4) (%)':<38} {'—':>10} {cen['nis_frac_joint']*100:>10.1f}")
    print(f"{'True joint radar+camera updates':<38} {'—':>10} {cen['joint_updates']:>10}")
    print(f"{'Scans with camera data':<38} {seq['scans_with_camera']:>10} {cen['scans_with_camera']:>10}")
    print("-" * 60)

    winner = "Sequential" if seq["rmse_m"] < cen["rmse_m"] else "Centralised"
    delta = abs(seq["rmse_m"] - cen["rmse_m"])

    def nis_label(value: float, borderline: float = 0.80) -> str:
        if np.isnan(value):
            return "N/A"
        if value >= 0.90:
            return "PASS ✓"
        if value >= borderline:
            return "BORDERLINE"
        return "FAIL ✗"

    print(f"\n  Lower RMSE                   : {winner} (Δ = {delta:.2f} m)")
    print(f"  Architecture difference      : {'CLEAR' if delta > 0.5 else 'NEGLIGIBLE'}")
    print(f"  NIS radar (seq)              : {nis_label(seq['nis_frac_radar'])}")
    print(f"  NIS camera (seq)             : {nis_label(seq['nis_frac_camera'])}")
    print(f"  NIS centralised overall      : {nis_label(cen['nis_frac_all'])}")
    print(f"  NIS centralised joint only   : {nis_label(cen['nis_frac_joint'], borderline=0.85)}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

def demo_fusion_tracker() -> RadarCameraFusionTracker:
    return RadarCameraFusionTracker.from_detection(
        _build_cfm(), range_m=800.0, bearing_rad=np.deg2rad(36.87)
    )


def _smoke_test() -> None:
    cfm = CoordinateFrameManager(
        camera_offset=np.array([-80.0, 120.0], dtype=float),
        radar_R=np.diag([25.0, np.deg2rad(0.3) ** 2]),
        camera_R=np.diag([64.0, np.deg2rad(0.15) ** 2]),
        ais_R=np.eye(2),
    )

    radar_meas = np.array([505.0, 0.21], dtype=float)
    camera_meas = np.array([420.0, 0.35], dtype=float)

    tracker = RadarCameraFusionTracker.from_detection(cfm, range_m=500.0, bearing_rad=0.2)
    tracker.predict(1.0)
    nis_seq = tracker.update_sequential(radar_meas, camera_meas)
    assert "radar" in nis_seq and "camera" in nis_seq

    tracker = RadarCameraFusionTracker.from_detection(cfm, range_m=500.0, bearing_rad=0.2)
    tracker.predict(1.0)
    nis_joint = tracker.update_centralised(radar_meas, camera_meas)
    assert nis_joint is not None and np.isfinite(nis_joint)

    tracker = RadarCameraFusionTracker.from_detection(cfm, range_m=500.0, bearing_rad=0.2)
    tracker.predict(1.0)
    nis_cam_only = tracker.update_centralised(radar_z=None, camera_z=camera_meas)
    assert nis_cam_only is not None and np.isfinite(nis_cam_only)

    print("T4 smoke test passed (including bearing-index fix).")


if __name__ == "__main__":
    _smoke_test()

    import sys
    if len(sys.argv) > 1:
        results = run_scenario_b(sys.argv[1])
        print_scenario_b_results(results)
    else:
        print("Usage: python T4_radar_camera_fusion.py harbour_sim_output/scenario_B.json")
