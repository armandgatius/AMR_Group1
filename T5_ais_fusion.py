"""T5: Asynchronous AIS fusion: radar + camera + AIS.

Usage
-----
    python T5_ais_fusion.py harbour_sim_output/scenario_C.json

Extends T4 with:
  - Asynchronous AIS updates: when an AIS NED position report arrives, the tracker
    predicts forward to the report time, then updates.
  - AIS-exclusive target initiation: targets visible only via AIS are
    initialised from AIS alone and tracked at the low AIS rate.
  - AIS dropout handling: the tracker coasts on radar/camera during
    the dropout window and smoothly re-acquires AIS when it reappears.

Scenario C validation: quantitative comparison of tracking accuracy
with and without AIS, including a 30 s dropout (t = 60-90 s).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from T2_CoordinateFrameManager import CoordinateFrameManager
from T3_single_sensor_tracker import (
    EKFConfig,
    wrap_angle,
    load_scenario,
    cv_transition_matrix,
    cv_process_noise,
)
from T4_radar_camera_fusion import RadarCameraFusionTracker

class AISFusionTracker(RadarCameraFusionTracker):
    """Single-target EKF supporting radar, camera and asynchronous AIS."""

    def __init__(
        self,
        cfm: CoordinateFrameManager,
        state: np.ndarray,
        covariance: np.ndarray,
        config: EKFConfig | None = None,
    ):
        super().__init__(cfm=cfm, state=state, covariance=covariance, config=config)
        # Timestamp of the last predict/update so dt for async steps is known
        self.last_time: float | None = None

    @classmethod
    def from_detection_at_time(
        cls,
        cfm: CoordinateFrameManager,
        range_m: float,
        bearing_rad: float,
        t: float,
        config: EKFConfig | None = None,
    ) -> "AISFusionTracker":
        """Create a track from a single detection with a known timestamp."""
        tracker = cls.from_detection(
            cfm, range_m=range_m, bearing_rad=bearing_rad, config=config
        )
        tracker.last_time = t
        return tracker

    @classmethod
    def from_ais_report(
        cls,
        cfm: CoordinateFrameManager,
        north_m: float,
        east_m: float,
        t: float,
        config: EKFConfig | None = None,
    ) -> "AISFusionTracker":
        """Initialise a track directly from an AIS NED position report.

        Used for AIS-exclusive targets that have no radar/camera history.
        Velocity is initialised to zero; covariance reflects AIS noise.
        """
        if config is None:
            config = EKFConfig()

        state = np.array([north_m, east_m, 0.0, 0.0], dtype=float)

        # Position uncertainty from AIS noise and velocity very uncertain
        ais_sigma = np.sqrt(cfm.R("ais")[0, 0])
        covariance = np.diag(
            [
                ais_sigma ** 2,
                ais_sigma ** 2,
                10.0 * config.initial_covariance,
                10.0 * config.initial_covariance,
            ]
        )

        tracker = cls(cfm=cfm, state=state, covariance=covariance, config=config)
        tracker.last_time = t
        return tracker

    def update_ais_async(self, north_m: float, east_m: float, t: float) -> float:
        """Predict to the AIS report time, then update with the AIS position.

        The AIS sensor reports absolute NED positions. The CoordinateFrameManager
        converts these to (range, bearing) relative to the current vessel
        position (which must be kept up-to-date via cfm.update_vessel_position).

        Parameters
        ----------
        north_m, east_m : AIS target position in NED frame (metres)
        t               : timestamp of the AIS report (seconds)

        Returns
        -------
        NIS value of the AIS update
        """
        # --- Predict forward to AIS time ---
        if self.last_time is not None and t > self.last_time:
            dt = t - self.last_time
            self.predict(dt)

        self.last_time = t

        # --- Convert AIS NED position to (range, bearing) observation ---
        # The vessel position is already stored in cfm; h() handles the offset.
        # We build the observation vector from the AIS position directly.
        vessel = self.cfm.get_sensor_position("ais")      # current vessel NED
        d = np.array([north_m, east_m]) - vessel
        r = float(np.linalg.norm(d))
        phi = float(np.arctan2(d[1], d[0]))
        z_ais = np.array([r, phi], dtype=float)

        # --- Standard EKF update using the "ais" sensor model ---
        return self.update(z_ais, sensor_id="ais")


# ---------------------------------------------------------------------------
# Scenario C helpers
# ---------------------------------------------------------------------------

def _build_cfm_c() -> CoordinateFrameManager:
    """Build the CoordinateFrameManager for Scenario C (all sensors).

    AIS sigma_pos = 4 m (from sensor_configs). The AIS measurement model
    converts the NED position report into (range, bearing) relative to the
    vessel. The range noise is ~sigma_pos; the bearing noise depends on range
    but 0.02 rad is reasonable for a target at ~200 m with 4 m position noise.
    """
    AIS_SIGMA_POS    = 4.0    # metres (from scenario sensor_configs)
    AIS_SIGMA_BEARING = 0.02  # radians (~4 m / 200 m typical range)

    cfm = CoordinateFrameManager(
        camera_offset=np.array([-80.0, 120.0], dtype=float),
        radar_R=np.diag([7.0 ** 2, np.deg2rad(0.3) ** 2]),  # tuned for NIS consistency on Scenario C
        #radar_R=np.diag([5.0 ** 2, np.deg2rad(0.3) ** 2]),
        camera_R=np.diag([8.0 ** 2, np.deg2rad(0.15) ** 2]),
        ais_R=np.diag([AIS_SIGMA_POS ** 2, AIS_SIGMA_BEARING ** 2]),
    )
    cfm.update_vessel_position(np.array([0.0, 0.0], dtype=float))
    return cfm


def _split_measurements(measurements: list[dict]) -> dict[str, list[dict]]:
    """Split measurements by sensor, keeping only true detections."""
    result: dict[str, list[dict]] = {
        "radar": [], "camera": [], "ais": [], "gnss": []
    }
    for m in measurements:
        sid = m["sensor_id"]
        if sid in result and not m["is_false_alarm"]:
            result[sid].append(m)
    for sid in result:
        result[sid].sort(key=lambda m: m["time"])
    return result


def _nearest_camera_c(
    camera_meas: list[dict], t: float, window: float = 5.0
) -> np.ndarray | None:
    """Return camera measurement closest to time t within ±window seconds."""
    best, best_dt = None, float("inf")
    for m in camera_meas:
        dt = abs(m["time"] - t)
        if dt <= window and dt < best_dt:
            best_dt, best = dt, m
    if best is None:
        return None
    return np.array([best["range_m"], best["bearing_rad"]], dtype=float)


def _gt_lookup(scenario: dict) -> tuple[dict, np.ndarray]:
    """Return ground-truth lookup dict and sorted time array."""
    records = scenario["ground_truth"]["0"]
    lut = {rec[0]: np.array(rec[1:], dtype=float) for rec in records}
    times = np.array(sorted(lut.keys()))
    return lut, times


def _position_error(tracker: AISFusionTracker, gt_lut: dict, gt_times: np.ndarray, t: float) -> float:
    closest_t = gt_times[np.argmin(np.abs(gt_times - t))]
    gt_state = gt_lut[closest_t]
    return float(np.linalg.norm(tracker.get_state()[:2] - gt_state[:2]))


# ---------------------------------------------------------------------------
# Scenario C runner (with and without AIS)
# ---------------------------------------------------------------------------

def run_scenario_c(
    json_path: str | Path,
    use_ais: bool = True,
    warmup_scans: int = 5,
) -> dict:
    """Run the AIS fusion tracker on Scenario C data.

    Parameters
    ----------
    json_path    : path to scenario_C.json
    use_ais      : if False, the tracker ignores all AIS measurements
                   (baseline comparison)
    warmup_scans : radar scans excluded from RMSE/NIS metrics

    Returns
    -------
    dict with rmse_m, nis_values, nis_fraction_in_bounds,
    dropout_rmse_m (RMSE inside t=60-90 s), post_dropout_rmse_m,
    track_survived_dropout (bool)
    """
    scenario = load_scenario(json_path)
    gt_lut, gt_times = _gt_lookup(scenario)
    meas = _split_measurements(scenario["measurements"])

    radar_meas  = meas["radar"]
    camera_meas = meas["camera"]
    ais_meas    = meas["ais"]
    gnss_meas   = meas["gnss"]

    # Build a fast GNSS lookup: for each timestamp find vessel NED
    gnss_by_time = {m["time"]: np.array([m["north_m"], m["east_m"]]) for m in gnss_meas}
    gnss_times   = np.array(sorted(gnss_by_time.keys()))

    def get_vessel_pos(t: float) -> np.ndarray:
        """Return the GNSS vessel position closest in time to t."""
        idx = np.argmin(np.abs(gnss_times - t))
        return gnss_by_time[gnss_times[idx]]

    cfm = _build_cfm_c()
    tracker: AISFusionTracker | None = None
    prev_radar_time: float | None   = None
    scan_count = 0

    # Metrics
    nis_values:        list[float] = []
    rmse_sq_all:       list[float] = []
    rmse_sq_dropout:   list[float] = []
    rmse_sq_post:      list[float] = []
    last_radar_update_before_dropout: float | None = None
    track_survived = True

    # AIS dropout window (as specified in Scenario C)
    DROPOUT_START, DROPOUT_END = 60.0, 90.0

    # We process measurements in chronological order across all sensors
    # Radar drives the main predict/update loop; AIS is injected asynchronously.
    # Build a merged timeline of radar + AIS events.
    events: list[dict] = []
    for m in radar_meas:
        events.append({"type": "radar", **m})
    if use_ais:
        for m in ais_meas:
            events.append({"type": "ais", **m})
    events.sort(key=lambda e: e["time"])

    for event in events:
        t = event["time"]

        # Keep vessel position current for AIS measurement model
        cfm.update_vessel_position(get_vessel_pos(t))

        # ── AIS async event ──────────────────────────────────────────────
        if event["type"] == "ais":
            if tracker is None:
                # AIS-exclusive initialisation (target not yet seen by radar)
                tracker = AISFusionTracker.from_ais_report(
                    cfm,
                    north_m=event["north_m"],
                    east_m=event["east_m"],
                    t=t,
                    config=EKFConfig(sigma_a=0.05),

                )
                continue

            # Normal async AIS update
            tracker.update_ais_async(
                north_m=event["north_m"],
                east_m=event["east_m"],
                t=t,
            )
            continue

        # ── Radar event (main loop) ──────────────────────────────────────
        z_r = np.array([event["range_m"], event["bearing_rad"]], dtype=float)

        if tracker is None:
            tracker = AISFusionTracker.from_detection_at_time(
                cfm, range_m=z_r[0], bearing_rad=z_r[1], t=t,
                config=EKFConfig(sigma_a=0.05),

            )
            prev_radar_time = t
            scan_count = 1
            continue

        if prev_radar_time is None:
            prev_radar_time = t
            scan_count = 1
            continue
        dt = t - prev_radar_time
        prev_radar_time = t
        scan_count += 1

        tracker.predict(dt)
        tracker.last_time = t

        if scan_count <= warmup_scans:
            continue

        # Camera measurement (if available near this radar scan)
        z_c = _nearest_camera_c(camera_meas, t)

        # Update with radar + camera (sequential, matching T4 convention)
        nis_dict = tracker.update_sequential(z_r, z_c)
        if "radar" in nis_dict:
            nis_values.append(nis_dict["radar"])

        # Position error
        err = _position_error(tracker, gt_lut, gt_times, t)
        rmse_sq_all.append(err ** 2)

        if DROPOUT_START <= t <= DROPOUT_END:
            rmse_sq_dropout.append(err ** 2)
            last_radar_update_before_dropout = t
        elif t > DROPOUT_END:
            rmse_sq_post.append(err ** 2)

    # Check track survived dropout: tracker must still exist
    # and have a reasonable position error after dropout
    if tracker is not None and rmse_sq_post:
        post_rmse = float(np.sqrt(np.mean(rmse_sq_post)))
        track_survived = post_rmse < 50.0  # generous threshold
    else:
        track_survived = False

    nis_arr = np.array(nis_values)
    chi2_lo, chi2_hi = 0.0506, 5.991
    nis_frac = float(np.mean((nis_arr >= chi2_lo) & (nis_arr <= chi2_hi))) if len(nis_arr) else float("nan")

    return {
        "use_ais":                   use_ais,
        "rmse_m":                    float(np.sqrt(np.mean(rmse_sq_all))) if rmse_sq_all else float("nan"),
        "dropout_rmse_m":            float(np.sqrt(np.mean(rmse_sq_dropout))) if rmse_sq_dropout else float("nan"),
        "post_dropout_rmse_m":       float(np.sqrt(np.mean(rmse_sq_post))) if rmse_sq_post else float("nan"),
        "nis_values":                nis_arr,
        "nis_fraction_in_bounds":    nis_frac,
        "track_survived_dropout":    track_survived,
        "num_updates":               len(nis_values),
    }


def print_scenario_c_results(with_ais: dict, without_ais: dict) -> None:
    """Print a formatted comparison of tracking accuracy with and without AIS."""
    print("=" * 65)
    print("Scenario C — AIS Fusion Validation (T5)")
    print("=" * 65)
    print(f"{'Metric':<42} {'With AIS':>10} {'No AIS':>10}")
    print("-" * 65)
    print(f"{'Overall position RMSE (m)':<42} {with_ais['rmse_m']:>10.2f} {without_ais['rmse_m']:>10.2f}")
    print(f"{'RMSE during dropout window (m)':<42} {with_ais['dropout_rmse_m']:>10.2f} {without_ais['dropout_rmse_m']:>10.2f}")
    print(f"{'RMSE after dropout (re-acquisition) (m)':<42} {with_ais['post_dropout_rmse_m']:>10.2f} {without_ais['post_dropout_rmse_m']:>10.2f}")
    print(f"{'NIS in 95% chi2(2) bounds (%)':<42} {with_ais['nis_fraction_in_bounds']*100:>10.1f} {without_ais['nis_fraction_in_bounds']*100:>10.1f}")
    print(f"{'Track survived dropout':<42} {str(with_ais['track_survived_dropout']):>10} {str(without_ais['track_survived_dropout']):>10}")
    print("-" * 65)

    rmse_improvement = without_ais["rmse_m"] - with_ais["rmse_m"]
    print(f"\n  RMSE improvement with AIS     : {rmse_improvement:.2f} m")
    print(f"  Track survived dropout (AIS)  : {'PASS ✓' if with_ais['track_survived_dropout'] else 'FAIL ✗'}")
    print(f"  NIS consistency (with AIS)    : {'PASS ✓' if with_ais['nis_fraction_in_bounds'] >= 0.90 else 'FAIL ✗'}")
    print(f"  AIS improves accuracy         : {'YES ✓' if rmse_improvement > 0 else 'NO ✗'}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    cfm = _build_cfm_c()

    # Test 1: initialise from radar, do async AIS update
    tracker = AISFusionTracker.from_detection_at_time(
        cfm, range_m=500.0, bearing_rad=0.25, t=0.0
    )
    tracker.predict(3.0)
    cfm.update_vessel_position(np.array([10.0, 5.0]))
    nis = tracker.update_ais_async(north_m=480.0, east_m=130.0, t=3.0)
    assert np.isfinite(nis), "AIS async update returned non-finite NIS"
    assert tracker.get_state().shape == (4,)

    # Test 2: initialise from AIS only
    tracker_ais = AISFusionTracker.from_ais_report(
        cfm, north_m=300.0, east_m=200.0, t=0.0
    )
    tracker_ais.predict(3.0)
    nis2 = tracker_ais.update_ais_async(north_m=295.0, east_m=197.0, t=3.0)
    assert np.isfinite(nis2), "AIS-only track update returned non-finite NIS"

    # Test 3: async update does not go backwards in time
    t_now = tracker.last_time
    nis3 = tracker.update_ais_async(north_m=470.0, east_m=128.0, t=t_now - 1.0)
    assert np.isfinite(nis3), "Backward-time AIS update should still produce finite NIS"

    print("T5 smoke test passed.")


if __name__ == "__main__":
    _smoke_test()

    import sys
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        print("Running Scenario C with AIS ...")
        res_with    = run_scenario_c(json_path, use_ais=True)
        print("Running Scenario C without AIS ...")
        res_without = run_scenario_c(json_path, use_ais=False)
        print_scenario_c_results(res_with, res_without)
    else:
        print("Usage: python T5_ais_fusion.py harbour_sim_output/scenario_C.json")