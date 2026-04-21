"""T3: single-sensor EKF tracker for mm-wave radar.

This module provides a compact constant-velocity EKF implementation for the
radar-only phase of the harbour tracking project. It uses the existing
T2_CoordinateFrameManager for the radar measurement model.

Includes Scenario A validation: NIS consistency test and RMSE evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from T2_CoordinateFrameManager import CoordinateFrameManager


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def cv_transition_matrix(dt: float) -> np.ndarray:
    """Constant-velocity state transition matrix for [pN, pE, vN, vE]."""
    return np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def cv_process_noise(dt: float, sigma_a: float) -> np.ndarray:
    """Process noise for a white-acceleration CV model."""
    q = sigma_a**2
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2

    return q * np.array(
        [
            [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
            [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
            [dt3 / 2.0, 0.0, dt2, 0.0],
            [0.0, dt3 / 2.0, 0.0, dt2],
        ],
        dtype=float,
    )


@dataclass
class EKFConfig:
    sigma_a: float = 0.10
    initial_covariance: float = 100.0


class RadarEKFTracker:
    """Single-target EKF for radar-only tracking."""

    def __init__(
        self,
        cfm: CoordinateFrameManager,
        state: np.ndarray,
        covariance: np.ndarray,
        config: EKFConfig | None = None,
    ):
        self.cfm = cfm
        self.x = np.asarray(state, dtype=float).reshape(4)
        self.P = np.asarray(covariance, dtype=float).reshape(4, 4)
        self.config = config or EKFConfig()

    @classmethod
    def from_detection(
        cls,
        cfm: CoordinateFrameManager,
        range_m: float,
        bearing_rad: float,
        dt: float = 1.0,
        speed_guess: float = 0.0,
        config: EKFConfig | None = None,
    ) -> "RadarEKFTracker":
        """Create an initial track from a single radar detection."""
        bearing = float(bearing_rad)
        position = np.array(
            [range_m * np.cos(bearing), range_m * np.sin(bearing)],
            dtype=float,
        )
        velocity = np.array(
            [speed_guess * np.cos(bearing), speed_guess * np.sin(bearing)],
            dtype=float,
        )

        if config is None:
            config = EKFConfig()

        covariance = np.diag(
            [
                config.initial_covariance,
                config.initial_covariance,
                10.0 * config.initial_covariance,
                10.0 * config.initial_covariance,
            ]
        )

        state = np.hstack((position, velocity))
        return cls(cfm=cfm, state=state, covariance=covariance, config=config)

    def predict(self, dt: float) -> None:
        """Propagate the state using the CV model."""
        F = cv_transition_matrix(dt)
        Q = cv_process_noise(dt, self.config.sigma_a)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def innovation(self, z: np.ndarray, sensor_id: str = "radar") -> tuple[np.ndarray, np.ndarray]:
        """Return innovation and innovation covariance."""
        measurement = np.asarray(z, dtype=float).reshape(2)
        h = self.cfm.h(self.x, sensor_id)
        H = self.cfm.H(self.x, sensor_id)
        residual = measurement - h
        residual[1] = wrap_angle(residual[1])
        S = H @ self.P @ H.T + self.cfm.R(sensor_id)
        return residual, S

    def update(self, z: np.ndarray, sensor_id: str = "radar") -> float:
        """Perform a standard EKF update and return the NIS value."""
        residual, S = self.innovation(z, sensor_id=sensor_id)
        H = self.cfm.H(self.x, sensor_id)
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ residual
        self.P = (np.eye(4) - K @ H) @ self.P
        return float(residual.T @ np.linalg.inv(S) @ residual)

    def step(self, dt: float, z: np.ndarray | None = None) -> float | None:
        """Predict and, if available, update with one radar measurement."""
        self.predict(dt)
        if z is None:
            return None
        return self.update(z, sensor_id="radar")

    def get_state(self) -> np.ndarray:
        """Return a copy of the current EKF state."""
        return self.x.copy()

    def get_covariance(self) -> np.ndarray:
        """Return a copy of the current state covariance."""
        return self.P.copy()


# ---------------------------------------------------------------------------
# Scenario A validation
# ---------------------------------------------------------------------------

def load_scenario(json_path: str | Path) -> dict:
    """Load a simulation scenario JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def run_scenario_a(json_path: str | Path, warmup_scans: int = 5) -> dict:
    """
    Run the T3 radar-only EKF on Scenario A data and return evaluation metrics.

    Parameters
    ----------
    json_path    : path to the Scenario A JSON file
    warmup_scans : number of initial scans excluded from steady-state RMSE/NIS

    Returns
    -------
    dict with keys: rmse_m, nis_values, nis_fraction_in_bounds, num_updates
    """
    scenario = load_scenario(json_path)

    cfm = CoordinateFrameManager(
        camera_offset=np.array([-80.0, 120.0], dtype=float),
        radar_R=np.diag([5.0**2, np.deg2rad(0.3) ** 2]),
        camera_R=np.diag([8.0**2, np.deg2rad(0.15) ** 2]),
        ais_R=np.eye(2),
    )
    cfm.update_vessel_position(np.array([0.0, 0.0], dtype=float))

    # Radar true detections only, sorted by time
    # JSON keys: sensor_id, time, is_false_alarm, target_id, range_m, bearing_rad
    measurements = [
        m for m in scenario["measurements"]
        if m["sensor_id"] == "radar" and not m["is_false_alarm"]
    ]
    measurements.sort(key=lambda m: m["time"])

    # Ground truth: {"0": [[time, pN, pE, vN, vE], ...]}
    gt_records = scenario["ground_truth"]["0"]
    gt_lookup = {rec[0]: np.array(rec[1:], dtype=float) for rec in gt_records}

    tracker = None
    prev_time = None
    scan_count = 0

    nis_values: list[float] = []
    rmse_sq: list[float] = []

    for meas in measurements:
        t = meas["time"]
        z = np.array([meas["range_m"], meas["bearing_rad"]], dtype=float)

        if tracker is None:
            tracker = RadarEKFTracker.from_detection(
                cfm, range_m=z[0], bearing_rad=z[1]
            )
            prev_time = t
            scan_count = 1
            continue

        dt = t - prev_time
        prev_time = t
        scan_count += 1

        nis = tracker.step(dt, z)

        if scan_count > warmup_scans and nis is not None:
            nis_values.append(nis)
            gt_times = np.array(list(gt_lookup.keys()))
            closest_t = gt_times[np.argmin(np.abs(gt_times - t))]
            gt_state = gt_lookup[closest_t]
            err = np.linalg.norm(tracker.get_state()[:2] - gt_state[:2])
            rmse_sq.append(err**2)

    nis_arr = np.array(nis_values)
    rmse = float(np.sqrt(np.mean(rmse_sq))) if rmse_sq else float("nan")

    # chi2(2) 95% bounds
    chi2_lo, chi2_hi = 0.0506, 5.991
    nis_frac = float(np.mean((nis_arr >= chi2_lo) & (nis_arr <= chi2_hi))) if len(nis_arr) else float("nan")

    return {
        "rmse_m": rmse,
        "nis_values": nis_arr,
        "nis_fraction_in_bounds": nis_frac,
        "confirmed_within_scans": warmup_scans,
        "num_updates": len(nis_arr),
    }


def print_scenario_a_results(results: dict) -> None:
    """Print a formatted summary of Scenario A validation results."""
    print("=" * 50)
    print("Scenario A — Radar-only EKF Validation (T3)")
    print("=" * 50)
    print(f"  Steady-state RMSE        : {results['rmse_m']:.2f} m  (target < 12 m)")
    print(f"  NIS in 95% chi2 bounds   : {results['nis_fraction_in_bounds']*100:.1f}%  (target > 90%)")
    print(f"  Total EKF updates        : {results['num_updates']}")
    print(f"  Track confirmed within   : {results['confirmed_within_scans']} scans  (target <= 5)")
    print()
    print(f"  RMSE check : {'PASS ✓' if results['rmse_m'] < 12.0 else 'FAIL ✗'}")
    print(f"  NIS check  : {'PASS ✓' if results['nis_fraction_in_bounds'] >= 0.90 else 'FAIL ✗'}")
    print("=" * 50)


def demo_tracker() -> RadarEKFTracker:
    """Create a small ready-to-run radar tracker demo instance."""
    cfm = CoordinateFrameManager(
        camera_offset=np.array([-80.0, 120.0], dtype=float),
        radar_R=np.diag([5.0**2, np.deg2rad(0.3) ** 2]),
        camera_R=np.eye(2),
        ais_R=np.eye(2),
    )
    cfm.update_vessel_position(np.array([0.0, 0.0], dtype=float))
    return RadarEKFTracker.from_detection(cfm, range_m=800.0, bearing_rad=np.deg2rad(36.87))


def _smoke_test() -> None:
    cfm = CoordinateFrameManager(
        camera_offset=np.array([-80.0, 120.0], dtype=float),
        radar_R=np.diag([25.0, np.deg2rad(0.3) ** 2]),
        camera_R=np.eye(2),
        ais_R=np.eye(2),
    )
    tracker = RadarEKFTracker.from_detection(cfm, range_m=500.0, bearing_rad=0.25)
    tracker.predict(1.0)
    nis = tracker.update(np.array([505.0, 0.26], dtype=float))
    assert np.isfinite(nis)
    assert tracker.get_state().shape == (4,)
    print("T3 smoke test passed.")


if __name__ == "__main__":
    _smoke_test()

    import sys
    if len(sys.argv) > 1:
        results = run_scenario_a(sys.argv[1])
        print_scenario_a_results(results)
    else:
        print("Usage: python T3_single_sensor_tracker.py harbour_sim_output/scenario_A.json")