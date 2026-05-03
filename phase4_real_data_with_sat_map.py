"""
Phase 4 — Real data validation.

Runs the complete tracking pipeline on the Copenhagen harbour dataset
(Dana IV departure, 5 March 2026).

Uses:
  - T2_CoordinateFrameManager
  - T6_gating_data_association.MultiTargetTracker through T7_Track_management.TrackManager
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import contextily as ctx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False
    print("WARNING: contextily not installed — satellite map will be skipped.")
    print("         Install with: pip install contextily")

from T2_CoordinateFrameManager import CoordinateFrameManager
from T7_Track_managment import TrackManager


RADAR_ROT  = np.deg2rad(-87.5)
CAMERA_ROT = np.deg2rad(28)

EVAL_T_END = 350.0

# NED origin — radar/camera installation position
ORIGIN_LAT = 55.69014690
ORIGIN_LON = 12.59998830


def ned_to_latlon(north_m, east_m):
    """Flat-earth NED → geographic (valid for distances < 50 km)."""
    lat = ORIGIN_LAT + np.rad2deg(north_m / 6_378_137.0)
    lon = ORIGIN_LON + np.rad2deg(east_m / (6_378_137.0 * np.cos(np.deg2rad(ORIGIN_LAT))))
    return lat, lon


print("Loading real data...")

radar  = pd.read_csv("Experimental data/mm_wave_radar.csv")
camera = pd.read_csv("Experimental data/camera.csv")
ais    = pd.read_csv("Experimental data/ais.csv")
gnss   = pd.read_csv("Experimental data/gnss.csv")

print(f"  Radar:  {len(radar)} detections")
print(f"  Camera: {len(camera)} detections")
print(f"  AIS:    {len(ais)} messages from {ais['mmsi'].nunique()} unique vessels")
print(f"  GNSS:   {len(gnss)} fixes")


# ---------------------------------------------------------------------
# Dana IV ground truth
# ---------------------------------------------------------------------

dana = ais[ais["mmsi"] == 219384000].copy().sort_values("time")

print(f"\nDana IV: {len(dana)} AIS points")

dana_times = dana["time"].values
dana_N     = dana["N"].values
dana_E     = dana["E"].values


def get_dana_pos(t, window=8.0):
    idx = int(np.argmin(np.abs(dana_times - t)))
    if abs(dana_times[idx] - t) <= window:
        return np.array([dana_N[idx], dana_E[idx]], dtype=float)
    return None


# ---------------------------------------------------------------------
# Own vessel GNSS lookup
# ---------------------------------------------------------------------

gnss_times = gnss["time"].values
gnss_N_arr = gnss["N"].values
gnss_E_arr = gnss["E"].values


def get_vessel(t):
    idx = int(np.argmin(np.abs(gnss_times - t)))
    return np.array([gnss_N_arr[idx], gnss_E_arr[idx]], dtype=float)


# ---------------------------------------------------------------------
# Coordinate frame manager
# ---------------------------------------------------------------------

cfm = CoordinateFrameManager(
    camera_offset=np.array([-80.0, 120.0], dtype=float),
    radar_R=np.diag([7.0**2, np.deg2rad(1.5) ** 2]),
    camera_R=np.diag([8.0**2, np.deg2rad(2.0) ** 2]),
    ais_R=np.diag([6.0**2, 6.0**2]),
)

cfm.update_vessel_position(np.array([0.0, 0.0]))

tm = TrackManager(cfm)


# ---------------------------------------------------------------------
# Main tracking loop
# ---------------------------------------------------------------------

track_history  = {}
vessel_history = []

rmse_sq_list = []
motp_list    = []
ce_list      = []
t_list       = []

radar_times   = sorted(radar["time"].unique())
used_ais_keys = set()   # deduplicate AIS: each message injected only once

print(f"\nProcessing {len(radar_times)} radar scans...")

for t in radar_times:
    vessel_pos = get_vessel(t)

    gnss_meas = [{
        "sensor_id": "gnss",
        "time":      float(t),
        "north_m":   float(vessel_pos[0]),
        "east_m":    float(vessel_pos[1]),
    }]

    vessel_history.append((t, vessel_pos[0], vessel_pos[1]))

    # ── Radar ────────────────────────────────────────────────────────
    radar_meas = []
    for _, row in radar[radar["time"] == t].iterrows():
        bearing = np.deg2rad(float(row["bearing"])) + RADAR_ROT
        radar_meas.append({
            "sensor_id":   "radar",
            "time":        float(t),
            "is_false_alarm": False,
            "range_m":     float(row["range"]),
            "bearing_rad": float(bearing),
        })

    # ── Camera (nearest scan within ±2 s) ────────────────────────────
    camera_meas = []
    cam_window = camera[
        (camera["time"] >= t - 2.0) & (camera["time"] <= t + 2.0)
    ]
    for _, row in cam_window.iterrows():
        cam_range   = float(np.sqrt(row["X"] ** 2 + row["Z"] ** 2))
        cam_bearing = float(np.arctan2(row["X"], row["Z"])) + CAMERA_ROT
        camera_meas.append({
            "sensor_id":   "camera",
            "time":        float(t),
            "is_false_alarm": False,
            "range_m":     cam_range,
            "bearing_rad": cam_bearing,
        })

    # ── AIS (deduplicated: each message icted only once) ───────────
    ais_meas = []
    ais_window = ais[
        (ais["time"] >= t - 4.0) & (ais["time"] <= t + 4.0)
    ]
    for _, row in ais_window.iterrows():
        key = (int(row["mmsi"]), float(row["time"]))
        if key in used_ais_keys:
            continue
        used_ais_keys.add(key)
        ais_meas.append({
            "sensor_id":   "ais",
            "time":        float(t),
            "is_false_alarm": False,
            "mmsi":        int(row["mmsi"]),
            "north_m":     float(row["N"]),
            "east_m":      float(row["E"]),
        })

    result = tm.step(float(t), radar_meas, camera_meas,
                     gnss_meas=gnss_meas, ais_meas=ais_meas)

    confirmed = tm.confirmed_tracks

    for tid, trk in confirmed.items():
        pos = trk.get_state()[:2]
        track_history.setdefault(tid, []).append((t, pos[0], pos[1]))

    if t <= EVAL_T_END:
        gt = get_dana_pos(t)
        t_list.append(t)
        ce_list.append(abs(len(confirmed) - 1))

        if gt is not None and confirmed:
            dists = [
                np.linalg.norm(trk.get_state()[:2] - gt)
                for trk in confirmed.values()
            ]
            best = min(dists)
            rmse_sq_list.append(best ** 2)
            motp_list.append(best)


# ---------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------

rmse_real   = float(np.sqrt(np.mean(rmse_sq_list))) if rmse_sq_list else float("nan")
motp_real   = float(np.mean(motp_list))             if motp_list    else float("nan")
ce_real     = float(np.mean(ce_list))               if ce_list      else float("nan")
long_tracks = [tid for tid, h in track_history.items() if len(h) >= 5]

print(f"\n{'=' * 55}")
print("Phase 4 — Real data metrics  (t ≤ 350 s, Dana IV in range)")
print(f"{'=' * 55}")
print(f"  RMSE vs Dana IV AIS        : {rmse_real:.1f} m")
print(f"  MOTP closest track         : {motp_real:.1f} m")
print(f"  Mean CE (|confirmed - 1|)  : {ce_real:.2f}")
print(f"  Max confirmed tracks       : {max(ce_list) + 1 if ce_list else 'N/A'}")
print(f"  Tracks with >=5 updates    : {len(long_tracks)}")
print(f"{'=' * 55}")
print("\nSimulation vs real performance:")
print("  Simulation MOTP Scen. D    : 3.9 m")
print(f"  Real data MOTP             : {motp_real:.1f} m")
print(f"  Gap                        : {motp_real - 3.9:.1f} m")
print("\nDominant causes of gap:")
print("  1. Sensor frame rotation not perfectly calibrated")
print("  2. Real AIS variance differs from simulation")
print("  3. Harbour multipath creates extra ghost detections")
print(f"{'=' * 55}")


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

print("\nGenerating trajectory plot...")

fig, ax = plt.subplots(figsize=(11, 11))

if HAS_CONTEXTILY:
    def to_plot(north_m, east_m):
        lat, lon = ned_to_latlon(north_m, east_m)
        return lon, lat   # x=lon, y=lat for map convention

    coord_label = ("Longitude", "Latitude")
    xlim = tuple(ned_to_latlon(0, v)[1] for v in (-600, 1500))
    ylim = tuple(ned_to_latlon(v, 0)[0] for v in (-600, 1500))
else:
    def to_plot(north_m, east_m):
        return east_m, north_m

    coord_label = ("East (m)", "North (m)")
    xlim = (-600, 1500)
    ylim = (-600, 1500)
    ax.set_facecolor("#F0F4F8")
    ax.grid(True, color="white", linewidth=1.2, zorder=0)

COLOURS = [
    "#E63946", "#2196F3", "#FF9800", "#9C27B0", "#00BCD4",
    "#4CAF50", "#FF5722", "#795548", "#607D8B", "#E91E63",
]

MIN_UPDATES = 5
plotted = 0

for tid, history in track_history.items():
    if len(history) < MIN_UPDATES:
        continue
    xs = [to_plot(h[1], h[2])[0] for h in history]
    ys = [to_plot(h[1], h[2])[1] for h in history]
    col = COLOURS[plotted % len(COLOURS)]
    ax.plot(xs, ys, color=col, linewidth=1.8, alpha=0.75, zorder=3)
    ax.scatter(xs[-1], ys[-1], color=col, s=50, zorder=4)
    ax.annotate(f"Track {tid}", (xs[-1], ys[-1]),
                fontsize=7, color=col, xytext=(5, 5),
                textcoords="offset points")
    plotted += 1

# Dana IV ground truth
dana_xs = [to_plot(n, e)[0] for n, e in zip(dana_N, dana_E)]
dana_ys = [to_plot(n, e)[1] for n, e in zip(dana_N, dana_E)]
ax.plot(dana_xs, dana_ys, "g--", linewidth=2.2, alpha=0.9,
        label="Dana IV AIS ground truth", zorder=5)
ax.scatter(dana_xs[0],  dana_ys[0],  color="green", marker="^",
           s=140, zorder=6, label="Dana IV start")
ax.scatter(dana_xs[-1], dana_ys[-1], color="green", marker="s",
           s=140, zorder=6, label="Dana IV end")

# Radar range circle (1 km)
theta = np.linspace(0, 2 * np.pi, 300)
circle_xs = [to_plot(1000 * np.cos(a), 1000 * np.sin(a))[0] for a in theta]
circle_ys = [to_plot(1000 * np.cos(a), 1000 * np.sin(a))[1] for a in theta]
ax.plot(circle_xs, circle_ys, "k--", linewidth=0.8, alpha=0.35,
        label="Radar range 1 km")

# Own vessel GNSS
v_arr = np.array(vessel_history)
vxs = [to_plot(n, e)[0] for n, e in zip(v_arr[:, 1], v_arr[:, 2])]
vys = [to_plot(n, e)[1] for n, e in zip(v_arr[:, 1], v_arr[:, 2])]
ax.plot(vxs, vys, color="navy", linewidth=1, linestyle="dotted",
        alpha=0.5, label="Own vessel GNSS")

origin_x, origin_y = to_plot(0, 0)
ax.scatter(origin_x, origin_y, color="black", s=220, marker="*",
           zorder=7, label="Radar / NED origin")

ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_xlabel(coord_label[0], fontsize=12)
ax.set_ylabel(coord_label[1], fontsize=12)
ax.set_title(
    "Copenhagen Harbour — Real Data Tracking\n"
    "Dana IV departure, 5 March 2026",
    fontsize=14, fontweight="bold",
)
ax.set_aspect("equal")

if HAS_CONTEXTILY:
    try:
        ctx.add_basemap(ax, crs="EPSG:4326",
                        source=ctx.providers.Esri.WorldImagery,
                        zoom=15, zorder=1)
    except Exception as e:
        print(f"WARNING: satellite tiles unavailable ({e})")

ax.legend(fontsize=9, loc="upper left")

stats = (
    f"Tracks >=5 updates: {plotted}\n"
    f"RMSE vs Dana IV: {rmse_real:.0f} m\n"
    f"MOTP t<={EVAL_T_END:.0f}s: {motp_real:.0f} m\n"
    f"Mean CE (|n_confirmed-1|): {ce_real:.2f}"
)

ax.text(0.02, 0.02, stats, transform=ax.transAxes, fontsize=9,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

plt.tight_layout()
plt.savefig("phase4_trajectories.png", dpi=150, bbox_inches="tight")
plt.show()

print("Plot saved as phase4_trajectories.png")
