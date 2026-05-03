'''
“If a target exists at position x, what would the sensor ideally measure?”

state: x=[p_N, p_E, v_N, v_E]
'''

import numpy as np

class CoordinateFrameManager:

    def __init__(self,
                 camera_offset: np.ndarray,
                 radar_R: np.ndarray,
                 camera_R: np.ndarray,
                 ais_R: np.ndarray):

        self.camera_offset = camera_offset
        self.radar_offset = np.array([0.0, 0.0])
        self.vessel_position = np.array([0.0, 0.0])

        self.R_dict = {
            "radar": radar_R,
            "camera": camera_R,
            "ais": ais_R
        }

    
    def update_vessel_position(self, vessel_ned: np.ndarray):
        self.vessel_position = vessel_ned.copy()


    def get_sensor_position(self, sensor_id: str) -> np.ndarray:
        if sensor_id == "radar":
            return self.radar_offset
        elif sensor_id == "camera":
            return self.camera_offset
        elif sensor_id == "ais":
            return self.vessel_position
        else:
            raise ValueError(f"Unknown sensor: {sensor_id}")

    
    def h(self, x: np.ndarray, sensor_id: str) -> np.ndarray:
        p = x[:2]
        s = self.get_sensor_position(sensor_id)

        d = p - s
        r = np.linalg.norm(d)
        phi = np.arctan2(d[1], d[0])

        return np.array([r, phi])

    
    def H(self, x: np.ndarray, sensor_id: str) -> np.ndarray:
        p = x[:2]
        s = self.get_sensor_position(sensor_id)

        dN = p[0] - s[0]
        dE = p[1] - s[1]

        r = np.sqrt(dN**2 + dE**2)

        if r < 1e-6:
            raise ValueError("Division by zero in Jacobian")

        H = np.zeros((2, 4))

        H[0, 0] = dN / r
        H[0, 1] = dE / r

        H[1, 0] = -dE / (r**2)
        H[1, 1] = dN / (r**2)

        return H

    
    def R(self, sensor_id: str) -> np.ndarray:
        return self.R_dict[sensor_id]

    def polar_to_ned(self, range_m: float, bearing_rad: float, sensor_id: str) -> np.ndarray:
        """Convert a (range, bearing) measurement to NED position."""
        s = self.get_sensor_position(sensor_id)
        return np.array([s[0] + range_m * np.cos(bearing_rad),
                          s[1] + range_m * np.sin(bearing_rad)], dtype=float)

    def R_cartesian(self, ned_pos: np.ndarray, sensor_id: str) -> np.ndarray:
        """Linearise the sensor's polar noise matrix into Cartesian (NED) space."""
        x = np.array([ned_pos[0], ned_pos[1], 0.0, 0.0])
        try:
            J     = self.H(x, sensor_id)[:, :2]
            J_inv = np.linalg.inv(J)
            return J_inv @ self.R(sensor_id) @ J_inv.T
        except (ValueError, np.linalg.LinAlgError):
            return np.eye(2) * 100.0

    @staticmethod
    def run_tests():
        print("Running enhanced CoordinateFrameManager tests...")

        cfm = CoordinateFrameManager(
            camera_offset=np.array([-80.0, 120.0]),
            radar_R=np.eye(2),
            camera_R=np.eye(2),
            ais_R=np.eye(2)
        )

        # ─────────────────────────────────────────────
        # TEST 1: Radar basic geometry (known triangle)
        # ─────────────────────────────────────────────
        x = np.array([3.0, 4.0, 0, 0])
        z = cfm.h(x, "radar")

        assert np.isclose(z[0], 5.0), "Radar range incorrect"
        assert np.isclose(z[1], np.arctan2(4.0, 3.0)), "Radar bearing incorrect"


        # ─────────────────────────────────────────────
        # TEST 2: Bearing sign convention (quadrant test)
        # ─────────────────────────────────────────────
        x = np.array([-10.0, 10.0, 0, 0])
        z = cfm.h(x, "radar")

        expected_phi = np.arctan2(10.0, -10.0)

        assert np.isclose(z[1], expected_phi), "Bearing quadrant incorrect"


        # ─────────────────────────────────────────────
        # TEST 3: Camera offset correctness
        # ─────────────────────────────────────────────
        cfm.camera_offset = np.array([10.0, 0.0])

        x = np.array([10.0, 10.0, 0, 0])  # target directly north of camera

        z = cfm.h(x, "camera")

        assert np.isclose(z[0], 10.0), "Camera range incorrect"
        assert np.isclose(z[1], np.pi / 2), "Camera bearing incorrect"


        # ─────────────────────────────────────────────
        # TEST 4: AIS equals radar when vessel at origin
        # ─────────────────────────────────────────────
        cfm.update_vessel_position(np.array([0.0, 0.0]))

        x = np.array([20.0, 0.0, 0, 0])

        z_ais = cfm.h(x, "ais")
        z_radar = cfm.h(x, "radar")

        assert np.allclose(z_ais, z_radar), "AIS vs radar mismatch at origin"


        # ─────────────────────────────────────────────
        # TEST 5: AIS moving reference correctness
        # ─────────────────────────────────────────────
        cfm.update_vessel_position(np.array([10.0, 5.0]))

        x = np.array([20.0, 15.0, 0, 0])

        z_ais = cfm.h(x, "ais")

        expected = np.array([10.0, 10.0])  # relative position
        z_exp = cfm.h(expected, "radar")

        assert np.allclose(z_ais, z_exp), "AIS moving reference incorrect"


        # ─────────────────────────────────────────────
        # TEST 6: Symmetry test (swap N/E should rotate bearing)
        # ─────────────────────────────────────────────
        x1 = np.array([10.0, 0.0, 0, 0])
        x2 = np.array([0.0, 10.0, 0, 0])

        z1 = cfm.h(x1, "radar")
        z2 = cfm.h(x2, "radar")

        assert np.isclose(z1[1], 0.0), "Bearing for east target incorrect"
        assert np.isclose(z2[1], np.pi/2), "Bearing for north target incorrect"


        # ─────────────────────────────────────────────
        # TEST 7: Jacobian shape
        # ─────────────────────────────────────────────
        x = np.array([50.0, 30.0, 0, 0])
        H = cfm.H(x, "radar")

        assert H.shape == (2, 4), "Jacobian shape incorrect"


        # ─────────────────────────────────────────────
        # TEST 8: Jacobian finite difference check (VERY IMPORTANT)
        # ─────────────────────────────────────────────
        eps = 1e-5
        x = np.array([20.0, 15.0, 0, 0])

        H_analytic = cfm.H(x, "radar")

        H_numeric = np.zeros_like(H_analytic)

        for i in range(2):  # only position affects measurement
            dx = np.zeros(4)
            dx[i] = eps

            h1 = cfm.h(x + dx, "radar")
            h0 = cfm.h(x - dx, "radar")

            H_numeric[:, i] = (h1 - h0) / (2 * eps)

        assert np.allclose(H_analytic[:, :2], H_numeric[:, :2], atol=1e-4), \
            "Jacobian mismatch with numerical derivative"


        # ─────────────────────────────────────────────
        # TEST 9: Division safety (edge case)
        # ─────────────────────────────────────────────
        x = np.array([0.0, 0.0, 0, 0])

        try:
            cfm.h(x, "radar")
        except Exception:
            assert False, "Radar failed at origin (should be handled or defined)"


        print("All enhanced tests passed!")


if __name__ == "__main__":
    CoordinateFrameManager.run_tests()

