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

    

    @staticmethod
    def run_tests():
        print("Running CoordinateFrameManager tests...")

        cfm = CoordinateFrameManager(
            camera_offset=np.array([-80.0, 120.0]),
            radar_R=np.eye(2),
            camera_R=np.eye(2),
            ais_R=np.eye(2)
        )

        # ── Test 1: Radar basic geometry ──────────
        x = np.array([40.0, 30.0, 0, 0])
        z = cfm.h(x, "radar")

        assert np.isclose(z[0], 50.0), "Radar range incorrect"
        assert np.isclose(z[1], 0.6435), "Radar bearing incorrect"

        # ── Test 2: Camera offset ────────────────
        cfm.camera_offset = np.array([0.0, 10.0])
        x = np.array([0.0, 20.0, 0, 0])

        z = cfm.h(x, "camera")

        assert np.isclose(z[0], 10.0), "Camera range incorrect"
        assert np.isclose(z[1], np.pi / 2), "Camera bearing incorrect"

        # ── Test 3: AIS consistency ──────────────
        # Make vessel coincide with radar
        cfm.update_vessel_position(np.array([0.0, 0.0]))

        z_ais = cfm.h(x, "ais")
        z_radar = cfm.h(x, "radar")

        assert np.allclose(z_ais, z_radar), "AIS conversion incorrect"

        # ── Test 4: Jacobian shape ───────────────
        x = np.array([50.0, 30.0, 0, 0])
        H = cfm.H(x, "radar")

        assert H.shape == (2, 4), "Jacobian shape incorrect"

        print("All tests passed!")


if __name__ == "__main__":
    CoordinateFrameManager.run_tests()

