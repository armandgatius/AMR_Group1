# Multi-Sensor Multi-Target Tracking System for Harbour Surveillance

## Installation

Python 3.10+ required.

```bash
pip install numpy==2.4.4 scipy==1.17.1 matplotlib pandas contextily
```

## How to run

**Simulated scenarios**

```bash
python T3_single_sensor_tracker.py harbour_sim_output/scenario_A.json  # Radar only
python T4_radar_camera_fusion.py   harbour_sim_output/scenario_B.json  # Radar + camera
python T5_ais_fusion.py            harbour_sim_output/scenario_C.json  # AIS fusion
python T6_gating_and_data_association.py harbour_sim_output/scenario_D.json  # Multi-target
python T7_Track_managment.py harbour_sim_output/scenario_D.json         # Scenario D
python T7_Track_managment.py harbour_sim_output/scenario_E.json         # Scenario E
```

**Real data**

```bash
python phase4_real_data_with_sat_map.py
```

## Description

This multi-target tracking system combines Mahalanobis-distance gating with Global Nearest Neighbour (GNN) data association to track multiple objects simultaneously using radar and camera measurements. For each track and sensor, measurements are first filtered using a statistical gate based on the Mahalanobis distance, where a detection is accepted if its distance to the predicted measurement is below a threshold derived from the χ² distribution with probability P=0.99. This step reduces the number of unlikely associations and improves robustness under clutter.

All gated measurements from all sensors are then combined into a single association problem. A cost matrix is constructed where each entry represents the Mahalanobis distance between a track and a measurement, and invalid associations (outside the gate) are assigned a large cost. The Hungarian algorithm is used to compute the optimal one-to-one assignment between tracks and measurements. Tracks that are not assigned receive a missed-detection flag, while unassigned measurements are considered for track initiation.

Track management includes initiation, confirmation, and deletion. Initiation is two-staged: an unassigned measurement first enters a pre-EKF tentative buffer that accumulates detections until a velocity estimate can be formed, at which point a full EKF track is initialised. New pre-tracks are only created if the measurement does not lie within the gate of a track already assigned in the same scan, and if it is not within a suppression radius of any existing EKF or confirmed track, which prevents spurious initiations from clutter and multipath. Tracks are confirmed using an M-of-N criterion: a track must receive at least M detections within the last N scans. For non-AIS tracks, confirmation additionally requires a minimum total displacement, age, and number of hits to avoid confirming stationary false alarms. Tracks are deleted after a number of consecutive missed detections. Confirmed tracks that show negligible total displacement over their lifetime are identified as static clutter and removed. Short-lived tracks in the vicinity of a well-established track are treated as multipath fragments and deleted. Finally, pairs of tracks with overlapping state estimates are merged, keeping the stronger track, to further reduce duplication.


The system is validated through several tests designed to assess gating performance, data association accuracy, and overall tracking quality in a multi-target scenario.

First, gating is validated by checking that each track gate contains at most one true detection and by computing an empirical gate inclusion rate. This verifies that the Mahalanobis gating is neither too strict (missing true detections) nor too loose (allowing excessive clutter).

Second, data association is evaluated by detecting two types of errors. A missed association occurs when a track is not assigned a measurement even though a true detection for that target exists in the scan. An identity swap occurs when a track is assigned a measurement originating from a different target, which is especially critical during the crossing interval. Identity swaps are counted to assess how well the tracker preserves target identity.

Third, the system is evaluated using the OSPA metric, which measures both localization accuracy and the correctness of the number of tracked targets. The mean OSPA is computed during the crossing period and after it, with the requirement that it remains below 40 m.

Finally, track-level performance is assessed by verifying that all targets are successfully tracked throughout the scenario. This includes checking that all tracks are confirmed, maintained during the crossing, and that the final position error of each track relative to the ground truth remains below a defined threshold.
