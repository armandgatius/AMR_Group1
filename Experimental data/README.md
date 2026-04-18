# Dataset Description

This dataset is associated with the "Multi-Sensor Multi-Target Tracking System for Harbour Surveillance" project
of the Autonomous Marine Robotics course 2026. The data correspond to the departure of Dana IV in the harbour of Copenhagen on 5 match 2026. 

# Important information

* there is 28deg of rotation between the camera frame and the NED frame

* there is 16deg of rotation between the radar frame and the NED frame 

* the variance of the GNSS and AIS measurement is 6m
* time is in seconds, distances in meters


* N : north position in NED frame

* E : east position in NED frame

* heading: in deg in the NED frame

* mmsi : indentification number of the ship

* ais_id : unique given local id for the ship

in camera.csv
* ID : visual ID of the ship (done by feature matching)

* Z & X : position in the camera frame

* sigma_x & sigma_z : uncertainty 

in mm_wave_radar.csv

* cluster_id : unique id for each detected object for a given time

* range & bearing : position in the radar frame (polar coordinates)

* covariance matrix is ((cov_range cov_range_bearing) (cov_range_bearing cov_bearing))

The lat/lon position of the camera and radar is : 55.69014690N / 12.59998830 E ,
which is also the position of the NED frame


