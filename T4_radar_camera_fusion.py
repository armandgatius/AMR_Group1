"""T4: radar + stereo-camera EKF fusion.

Extends the T3 radar-only EKF with:
- sequential fusion: radar update, then camera update
- centralised fusion: one joint EKF update with stacked measurements

Scenario B validation compares RMSE and NIS consistency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from CoordinateFrameManager import CoordinateFrameManager
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

def _build_cfm() -> CoordinateFrameManager:
    cfm = CoordinateFrameManager(
        camera_offset=np.array([-80.0, 120.0], dtype=float),
        radar_R=np.diag([6.0**2, np.deg2rad(0.35) ** 2]),
        camera_R=np.diag([8.0**2, np.deg2rad(0.15) ** 2]),
        ais_R=np.eye(2),
    )
    cfm.update_vessel_position(np.array([0.0, 0.0], dtype=float))
    return cfm


def _split_by_sensor(measurements: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return true radar and true camera measurements, sorted by time."""
    radar = sorted(
        [m for m in measurements if m["sensor_id"] == "radar" and not m["is_false_alarm"]],
        key=lambda m: m["time"],
    )
    camera = sorted(
        [m for m in measurements if m["sensor_id"] == "camera" and not m["is_false_alarm"]],
        key=lambda m: m["time"],
    )
    return radar, camera


def _nearest_camera(
    camera_meas: list[dict],
    t: float,
    window: float = 5.0,
) -> np.ndarray | None:
    """Return the camera measurement closest to time t within ±window."""
    best = None
    best_dt = float("inf")

    for m in camera_meas:
        dt = abs(m["time"] - t)
        if dt <= window and dt < best_dt:
            best_dt = dt
            best = m

    if best is None:
        return None

    return np.array([best["range_m"], best["bearing_rad"]], dtype=float)


def run_scenario_b(json_path: str | Path, warmup_scans: int = 0) -> dict:
    """
    Run both fusion architectures on Scenario B.

    As in T3, the first radar scan is used only for initialisation.
    Metrics are collected from subsequent radar-driven updates.
    """
    scenario = load_scenario(json_path)

    gt_records = scenario["ground_truth"]["0"]
    gt_lookup = {rec[0]: np.array(rec[1:], dtype=float) for rec in gt_records}
    gt_times = np.array(sorted(gt_lookup.keys()))

    radar_meas, camera_meas = _split_by_sensor(scenario["measurements"])

    chi2_lo2, chi2_hi2 = 0.0506, 5.991
    chi2_lo4, chi2_hi4 = 0.711, 9.488

    results = {}

    for architecture in ("sequential", "centralised"):
        cfm = _build_cfm()
        tracker: RadarCameraFusionTracker | None = None
        prev_time: float | None = None
        scan_count = 0

        rmse_sq: list[float] = []
        nis_radar: list[float] = []
        nis_camera: list[float] = []
        nis_joint: list[float] = []
        scans_with_camera = 0

        for meas in radar_meas:
            t = meas["time"]
            z_r = np.array([meas["range_m"], meas["bearing_rad"]], dtype=float)
            z_c = _nearest_camera(camera_meas, t)

            if tracker is None:
                tracker = RadarCameraFusionTracker.from_detection(
                    cfm, range_m=z_r[0], bearing_rad=z_r[1]
                )
                prev_time = t
                scan_count = 1
                continue

            dt = t - prev_time
            prev_time = t
            scan_count += 1

            tracker.predict(dt)

            # Match T3 convention exactly: exclude first `warmup_scans`
            # updates after initialisation.
            if (scan_count - 2) < warmup_scans:
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
                    nis_joint.append(nis)
                if z_c is not None:
                    scans_with_camera += 1

            closest_t = gt_times[np.argmin(np.abs(gt_times - t))]
            gt_state = gt_lookup[closest_t]
            err = np.linalg.norm(tracker.get_state()[:2] - gt_state[:2])
            rmse_sq.append(err**2)

        rmse = float(np.sqrt(np.mean(rmse_sq))) if rmse_sq else float("nan")

        if architecture == "sequential":
            nr = np.array(nis_radar)
            nc = np.array(nis_camera)
            results["sequential"] = {
                "rmse_m": rmse,
                "nis_frac_radar": float(np.mean((nr >= chi2_lo2) & (nr <= chi2_hi2))) if len(nr) else float("nan"),
                "nis_frac_camera": float(np.mean((nc >= chi2_lo2) & (nc <= chi2_hi2))) if len(nc) else float("nan"),
                "scans_with_camera": scans_with_camera,
            }
        else:
            nj = np.array(nis_joint)
            results["centralised"] = {
                "rmse_m": rmse,
                "nis_frac_joint": float(np.mean((nj >= chi2_lo4) & (nj <= chi2_hi4))) if len(nj) else float("nan"),
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
    print(f"{'NIS in bounds — joint chi2(4) (%)':<38} {'—':>10} {cen['nis_frac_joint']*100:>10.1f}")
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
    print(f"  NIS joint (cen)              : {nis_label(cen['nis_frac_joint'])}")
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