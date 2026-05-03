"""T7: Track management — full track lifecycle for multi-target tracking.

Usage
-----
    python T7_track_management.py harbour_sim_output/scenario_D.json
    python T7_track_management.py harbour_sim_output/scenario_E.json

Implements the complete track lifecycle on top of T6 (gating & data association):
  - Tentative : a new EKF is spawned for each unmatched detection
  - Confirmed : M-of-N confirmation logic (default M=3, N=5, configurable)
  - Coasting  : predict-only EKF steps on missed detections; gate widens
  - Deleted   : after K_del consecutive missed detections (configurable)
  - Duplicate merging via Mahalanobis distance between track state estimates

Validation metrics
------------------
  MOTP : Multiple Object Tracking Precision — mean localisation error over
         all matched confirmed-track / true-target pairs (min-distance assignment)
  CE   : Cardinality Error — mean |num_confirmed_tracks − num_active_targets|
         reported as time series and scalar average
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from T2_CoordinateFrameManager import CoordinateFrameManager
from T3_single_sensor_tracker import load_scenario, EKFConfig
from T5_ais_fusion import AISFusionTracker
from T6_gating_and_data_association import _split_measurements


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrackManagerConfig:
    """All tunable parameters for the track lifecycle."""
    M: int   = 2      # confirmations needed within N scans
    N: int   = 3      # window for M-of-N confirmation
    K_del: int = 6    # consecutive missed detections before deletion
    gate_threshold: float = 7.815   # chi2(2) at 97.5% — slightly wider than T6
    merge_threshold: float = 3.0    # Mahalanobis distance for duplicate merge
    sigma_a: float = 0.10           # process noise (m/s²)
    initial_covariance: float = 100.0


# ---------------------------------------------------------------------------
# Track states
# ---------------------------------------------------------------------------

class TrackStatus:
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    COASTING  = "coasting"
    DELETED   = "deleted"


# ---------------------------------------------------------------------------
# Managed track
# ---------------------------------------------------------------------------

_track_counter = 0

class ManagedTrack:
    """One track with full lifecycle bookkeeping."""

    def __init__(
        self,
        tracker: AISFusionTracker,
        cfm: CoordinateFrameManager,
        config: TrackManagerConfig,
        t: float,
    ):
        global _track_counter
        _track_counter += 1
        self.track_id: int = _track_counter

        self.tracker   = tracker
        self.cfm       = cfm
        self.config    = config
        self.status    = TrackStatus.TENTATIVE

        self.age: int            = 0    # total scans since birth
        self.hit_history: list   = []   # 1=hit, 0=miss, rolling window N
        self.consecutive_misses: int = 0
        self.t_last: float       = t

    # ── state accessors ──────────────────────────────────────────────────

    def get_state(self) -> np.ndarray:
        return self.tracker.get_state()

    def get_covariance(self) -> np.ndarray:
        return self.tracker.get_covariance()

    def position(self) -> np.ndarray:
        return self.get_state()[:2]

    # ── gate ──────────────────────────────────────────────────────────────

    def mahalanobis(self, z: np.ndarray, sensor_id: str) -> float:
        """Mahalanobis distance between measurement z and this track."""
        x   = self.get_state()
        H   = self.cfm.H(x, sensor_id)
        P   = self.get_covariance()
        R   = self.cfm.R(sensor_id)
        h   = self.cfm.h(x, sensor_id)
        S   = H @ P @ H.T + R

        if np.linalg.det(S) < 1e-12:
            return 1e9

        v = z - h
        v[1] = np.arctan2(np.sin(v[1]), np.cos(v[1]))   # wrap bearing
        return float(v.T @ np.linalg.inv(S) @ v)

    def in_gate(self, z: np.ndarray, sensor_id: str) -> bool:
        return self.mahalanobis(z, sensor_id) <= self.config.gate_threshold

    # ── lifecycle updates ─────────────────────────────────────────────────

    def predict(self, dt: float) -> None:
        self.tracker.predict(dt)

    def register_hit(self, z: np.ndarray, sensor_id: str, t: float) -> None:
        """Record an associated detection and update the EKF."""
        self.tracker.update(z, sensor_id)
        self.age += 1
        self.consecutive_misses = 0
        self.hit_history.append(1)
        if len(self.hit_history) > self.config.N:
            self.hit_history.pop(0)
        self.t_last = t
        self._update_status()

    def register_miss(self) -> None:
        """Record a missed detection (coast)."""
        self.age += 1
        self.consecutive_misses += 1
        self.hit_history.append(0)
        if len(self.hit_history) > self.config.N:
            self.hit_history.pop(0)
        self._update_status()

    def _update_status(self) -> None:
        """Transition between track states based on hit/miss history."""
        if self.status == TrackStatus.DELETED:
            return

        # Deletion check — highest priority
        if self.consecutive_misses >= self.config.K_del:
            self.status = TrackStatus.DELETED
            return

        hits_in_window = sum(self.hit_history[-self.config.N:])

        if self.status == TrackStatus.TENTATIVE:
            if hits_in_window >= self.config.M:
                self.status = TrackStatus.CONFIRMED
            elif self.consecutive_misses >= self.config.K_del:
                self.status = TrackStatus.DELETED

        elif self.status == TrackStatus.CONFIRMED:
            if self.consecutive_misses > 0:
                self.status = TrackStatus.COASTING

        elif self.status == TrackStatus.COASTING:
            if self.consecutive_misses == 0:
                self.status = TrackStatus.CONFIRMED
            elif self.consecutive_misses >= self.config.K_del:
                self.status = TrackStatus.DELETED

    @property
    def is_active(self) -> bool:
        return self.status != TrackStatus.DELETED

    @property
    def is_confirmed(self) -> bool:
        return self.status == TrackStatus.CONFIRMED


# ---------------------------------------------------------------------------
# Track manager
# ---------------------------------------------------------------------------

class TrackManager:
    """Full multi-target track lifecycle manager.

    Usage
    -----
    tm = TrackManager(cfm)
    for each timestep:
        tm.update(detections_by_sensor, t)
    confirmed = tm.confirmed_tracks
    """

    def __init__(
        self,
        cfm: CoordinateFrameManager,
        config: TrackManagerConfig | None = None,
    ):
        self.cfm    = cfm
        self.config = config or TrackManagerConfig()
        self.tracks: List[ManagedTrack] = []

        # For metrics
        self._motp_sum:   float = 0.0
        self._motp_count: int   = 0
        self._ce_series:  List[float] = []
        self._t_series:   List[float] = []

    # ── main update ──────────────────────────────────────────────────────

    def update(
        self,
        detections_by_sensor: Dict[str, List[dict]],
        t: float,
        n_true_targets: int | None = None,
    ) -> None:
        """Process one timestep: predict all tracks, associate, update lifecycle.

        Parameters
        ----------
        detections_by_sensor : dict mapping sensor_id → list of measurement dicts
        t                    : current timestamp (seconds)
        n_true_targets       : ground-truth target count at this time (for CE)
        """
        active = [tr for tr in self.tracks if tr.is_active]

        # ── 1. Predict all active tracks ─────────────────────────────────
        for tr in active:
            dt = t - tr.t_last
            if dt > 0:
                tr.predict(dt)
            tr.t_last = t

        # ── 2. Gate & associate — one sensor at a time (sequential) ──────
        # Build a set of track indices that got at least one hit this step
        hit_tracks: set = set()
        unmatched_dets: List[Tuple[np.ndarray, str]] = []  # (z, sensor_id)

        for sensor_id, dets in detections_by_sensor.items():
            if not dets:
                continue
            if sensor_id not in ("radar", "camera", "ais"):
                continue

            # Build cost matrix: rows = tracks, cols = detections
            z_list = []
            for d in dets:
                if sensor_id in ("radar", "camera"):
                    if d.get("range_m") is None or d.get("bearing_rad") is None:
                        continue
                    if not (np.isfinite(d["range_m"]) and np.isfinite(d["bearing_rad"])):
                        continue
                    z_list.append(np.array([d["range_m"], d["bearing_rad"]], dtype=float))
                elif sensor_id == "ais":
                    if d.get("north_m") is None or d.get("east_m") is None:
                        continue
                    # Convert AIS NED position to (range, bearing) relative to vessel
                    vessel = self.cfm.get_sensor_position("ais")
                    delta  = np.array([d["north_m"], d["east_m"]]) - vessel
                    r      = float(np.linalg.norm(delta))
                    phi    = float(np.arctan2(delta[1], delta[0]))
                    z_list.append(np.array([r, phi], dtype=float))

            if not z_list:
                continue

            n_tr = len(active)
            n_det = len(z_list)
            cost = np.full((n_tr, n_det), 1e9)

            for i, tr in enumerate(active):
                for j, z in enumerate(z_list):
                    d2 = tr.mahalanobis(z, sensor_id)
                    if d2 <= self.config.gate_threshold:
                        cost[i, j] = d2

            # Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(cost)
            assigned_det_idx = set()

            for i, j in zip(row_ind, col_ind):
                if cost[i, j] >= 1e9:
                    continue
                active[i].register_hit(z_list[j], sensor_id, t)
                hit_tracks.add(i)
                assigned_det_idx.add(j)

            # Collect unmatched detections for track initiation
            for j, z in enumerate(z_list):
                if j not in assigned_det_idx:
                    unmatched_dets.append((z, sensor_id, dets[j]))

        # ── 3. Missed-detection flags for tracks with no hit ─────────────
        for i, tr in enumerate(active):
            if i not in hit_tracks:
                tr.register_miss()

        # ── 4. Initiate new tentative tracks from unmatched detections ────
        for item in unmatched_dets:
            z, sensor_id, raw = item
            ekf_cfg = EKFConfig(
                sigma_a=self.config.sigma_a,
                initial_covariance=self.config.initial_covariance,
            )
            new_tracker = AISFusionTracker.from_detection_at_time(
                self.cfm,
                range_m=float(z[0]),
                bearing_rad=float(z[1]),
                t=t,
                config=ekf_cfg,
            )
            self.tracks.append(ManagedTrack(new_tracker, self.cfm, self.config, t))

        # ── 5. Merge duplicate confirmed tracks ───────────────────────────
        self._merge_duplicates()

        # ── 6. Remove deleted tracks ──────────────────────────────────────
        self.tracks = [tr for tr in self.tracks if tr.is_active]

        # ── 7. Record CE metric ───────────────────────────────────────────
        if n_true_targets is not None:
            n_conf = len(self.confirmed_tracks)
            self._ce_series.append(abs(n_conf - n_true_targets))
            self._t_series.append(t)

    # ── duplicate merging ─────────────────────────────────────────────────

    def _merge_duplicates(self) -> None:
        """Merge confirmed tracks that are very close in state space."""
        confirmed = [tr for tr in self.tracks if tr.is_confirmed]
        to_delete: set = set()

        for i in range(len(confirmed)):
            if id(confirmed[i]) in to_delete:
                continue
            for j in range(i + 1, len(confirmed)):
                if id(confirmed[j]) in to_delete:
                    continue
                d2 = self._track_mahalanobis(confirmed[i], confirmed[j])
                if d2 < self.config.merge_threshold:
                    # Keep the older track (smaller track_id), delete the newer
                    to_delete.add(id(confirmed[j]))

        self.tracks = [tr for tr in self.tracks if id(tr) not in to_delete]

    def _track_mahalanobis(self, tr_a: ManagedTrack, tr_b: ManagedTrack) -> float:
        """Mahalanobis distance between two track state estimates."""
        dx = tr_a.position() - tr_b.position()
        S  = tr_a.get_covariance()[:2, :2] + tr_b.get_covariance()[:2, :2]
        if np.linalg.det(S) < 1e-12:
            return float(np.linalg.norm(dx))
        return float(dx.T @ np.linalg.inv(S) @ dx)

    # ── properties ────────────────────────────────────────────────────────

    @property
    def confirmed_tracks(self) -> List[ManagedTrack]:
        return [tr for tr in self.tracks if tr.is_confirmed]

    # ── metrics ───────────────────────────────────────────────────────────

    def compute_motp(
        self,
        gt_positions: np.ndarray,
    ) -> float:
        """Compute MOTP for the current confirmed tracks vs ground-truth positions.

        Parameters
        ----------
        gt_positions : (K, 2) array of true target NED positions [N, E]

        Returns
        -------
        Mean localisation error in metres (lower is better).
        """
        conf = self.confirmed_tracks
        if not conf or len(gt_positions) == 0:
            return float("nan")

        n_tr = len(conf)
        n_gt = len(gt_positions)
        cost = np.zeros((n_tr, n_gt))

        for i, tr in enumerate(conf):
            for j, gt in enumerate(gt_positions):
                cost[i, j] = np.linalg.norm(tr.position() - gt[:2])

        row_ind, col_ind = linear_sum_assignment(cost)
        return float(np.mean([cost[i, j] for i, j in zip(row_ind, col_ind)]))

    def ce_scalar(self) -> float:
        """Mean cardinality error over all recorded timesteps."""
        if not self._ce_series:
            return float("nan")
        return float(np.mean(self._ce_series))

    def ce_time_series(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (times, CE values) arrays."""
        return np.array(self._t_series), np.array(self._ce_series)


# ---------------------------------------------------------------------------
# CFM builder (shared with T6)
# ---------------------------------------------------------------------------

def _build_cfm() -> CoordinateFrameManager:
    return CoordinateFrameManager(
        camera_offset=np.array([-80.0, 120.0], dtype=float),
        radar_R=np.diag([5.0 ** 2, np.deg2rad(0.3) ** 2]),
        camera_R=np.diag([8.0 ** 2, np.deg2rad(0.15) ** 2]),
        ais_R=np.diag([4.0 ** 2, 0.02 ** 2]),
    )


# ---------------------------------------------------------------------------
# Ground-truth helpers
# ---------------------------------------------------------------------------

def _active_gt_positions(scenario: dict, t: float, tol: float = 0.5) -> np.ndarray:
    """Return (K, 2) array of active ground-truth NED positions at time t."""
    positions = []
    for target_id, records in scenario["ground_truth"].items():
        for rec in records:
            if abs(rec[0] - t) <= tol:
                state = np.array(rec[1:], dtype=float)
                if np.isfinite(state).all():
                    positions.append(state[:2])
                break
    return np.array(positions) if positions else np.empty((0, 2))


def _n_active_targets(scenario: dict, t: float, tol: float = 0.5) -> int:
    return len(_active_gt_positions(scenario, t, tol))


# ---------------------------------------------------------------------------
# Scenario runners
# ---------------------------------------------------------------------------

def run_scenario(
    json_path: str | Path,
    n_expected_targets: int,
    config: TrackManagerConfig | None = None,
    warmup_scans: int = 4,
) -> dict:
    """Run the full track management pipeline on any scenario JSON.

    Parameters
    ----------
    json_path           : path to scenario JSON file
    n_expected_targets  : expected number of targets (4 for D, 6 for E)
    config              : TrackManagerConfig (uses defaults if None)

    Returns
    -------
    dict with motp_m, ce_scalar, ce_series, t_series,
    n_confirmed_final, track_survived (bool)
    """
    scenario = load_scenario(json_path)
    cfm = _build_cfm()
    cfm.update_vessel_position(np.array([0.0, 0.0], dtype=float))

    # Update vessel position from GNSS
    gnss_meas = [m for m in scenario["measurements"] if m["sensor_id"] == "gnss"]
    gnss_by_t = {m["time"]: np.array([m["north_m"], m["east_m"]]) for m in gnss_meas}
    gnss_times = np.array(sorted(gnss_by_t.keys()))

    def get_vessel(t: float) -> np.ndarray:
        if len(gnss_times) == 0:
            return np.array([0.0, 0.0])
        idx = np.argmin(np.abs(gnss_times - t))
        return gnss_by_t[gnss_times[idx]]

    cfg = config or TrackManagerConfig()
    tm  = TrackManager(cfm, cfg)

    # Build sorted radar scan times (main clock)
    radar_times = sorted({
        m["time"] for m in scenario["measurements"]
        if m["sensor_id"] == "radar"
    })

    # Split all measurements
    meas = _split_measurements(scenario["measurements"])

    motp_sum   = 0.0
    motp_count = 0
    scan_idx   = 0

    for t in radar_times:
        scan_idx += 1
        cfm.update_vessel_position(get_vessel(t))

        # Collect detections at this timestep (±window for each sensor)
        def dets_at(sensor_id: str, window: float = 0.5) -> List[dict]:
            return [m for m in meas[sensor_id] if abs(m["time"] - t) <= window]

        detections = {
            "radar":  dets_at("radar",  0.1),
            "camera": dets_at("camera", 0.5),
            "ais":    dets_at("ais",    1.5),
        }

        # Skip CE during warmup, tracks haven't confirmed yet so CE would
        # artificially spike and skew the average
        n_true = _n_active_targets(scenario, t) if scan_idx > warmup_scans else None
        tm.update(detections, t, n_true_targets=n_true)

        # MOTP at this step
        gt_pos = _active_gt_positions(scenario, t)
        if len(gt_pos) > 0 and tm.confirmed_tracks:
            motp_step = tm.compute_motp(gt_pos)
            if np.isfinite(motp_step):
                motp_sum   += motp_step
                motp_count += 1

    motp = motp_sum / motp_count if motp_count > 0 else float("nan")
    ce   = tm.ce_scalar()
    t_ser, ce_ser = tm.ce_time_series()

    return {
        "motp_m":           motp,
        "ce_scalar":        ce,
        "ce_series":        ce_ser,
        "t_series":         t_ser,
        "n_confirmed_final": len(tm.confirmed_tracks),
        "n_expected":       n_expected_targets,
        "all_found":        len(tm.confirmed_tracks) >= n_expected_targets,
        "track_manager":    tm,
    }


def print_results(label: str, res: dict, motp_target: float, ce_target: float) -> None:
    print("=" * 55)
    print(f"{label}")
    print("=" * 55)
    print(f"  Confirmed tracks (final)  : {res['n_confirmed_final']}  (expected {res['n_expected']})")
    print(f"  MOTP                      : {res['motp_m']:.2f} m  (target < {motp_target} m)")
    print(f"  Cardinality Error (CE)    : {res['ce_scalar']:.2f}  (target < {ce_target})")
    print()
    motp_ok = res["motp_m"] < motp_target
    ce_ok   = res["ce_scalar"] < ce_target
    print(f"  MOTP check : {'PASS ✓' if motp_ok else 'FAIL ✗'}")
    print(f"  CE check   : {'PASS ✓' if ce_ok   else 'FAIL ✗'}")
    print(f"  Overall    : {'PASS ✓' if motp_ok and ce_ok else 'FAIL ✗'}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    cfm = _build_cfm()
    cfm.update_vessel_position(np.array([0.0, 0.0]))
    cfg = TrackManagerConfig(M=2, N=3, K_del=3)
    tm  = TrackManager(cfm, cfg)

    # Simulate 10 radar detections for 2 targets
    for step in range(10):
        t = float(step) * 3.3
        dets = {
            "radar": [
                {"sensor_id": "radar", "time": t, "is_false_alarm": False,
                 "range_m": 500.0 + step * 2, "bearing_rad": 0.5},
                {"sensor_id": "radar", "time": t, "is_false_alarm": False,
                 "range_m": 300.0 + step * 1, "bearing_rad": -0.3},
            ],
            "camera": [], "ais": [],
        }
        tm.update(dets, t, n_true_targets=2)

    conf = tm.confirmed_tracks
    assert len(conf) >= 1, f"Expected confirmed tracks, got {len(conf)}"
    assert all(tr.status == TrackStatus.CONFIRMED for tr in conf)
    assert tm.ce_scalar() >= 0.0
    print(f"T7 smoke test passed. ({len(conf)} confirmed tracks after 10 steps)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _smoke_test()

    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        name = path.stem.upper()

        if "D" in name:
            print(f"\nRunning Scenario D ...")
            res = run_scenario(path, n_expected_targets=4)
            print_results("Scenario D — Multi-target, crossing trajectories (T7)", res, motp_target=15.0, ce_target=0.5)
        elif "E" in name:
            print(f"\nRunning Scenario E ...")
            res = run_scenario(path, n_expected_targets=6)
            print_results("Scenario E — Mixed AIS/non-AIS harbour traffic (T7)", res, motp_target=20.0, ce_target=1.0)
        else:
            print("Running scenario ...")
            res = run_scenario(path, n_expected_targets=4)
            print_results(f"{name} results", res, motp_target=20.0, ce_target=1.0)
    else:
        print("Usage:")
        print("  python T7_track_management.py harbour_sim_output/scenario_D.json")
        print("  python T7_track_management.py harbour_sim_output/scenario_E.json")