import unittest

from T4_radar_camera_fusion import run_scenario_b


class ScenarioBT4RegressionTest(unittest.TestCase):
    def test_event_driven_fusion_is_nis_consistent(self):
        results = run_scenario_b("harbour_sim_output/scenario_B.json")

        sequential = results["sequential"]
        centralised = results["centralised"]

        self.assertLess(sequential["rmse_m"], 4.0)
        self.assertGreaterEqual(sequential["nis_frac_radar"], 0.90)
        self.assertGreaterEqual(sequential["nis_frac_camera"], 0.90)
        self.assertLess(centralised["rmse_m"], 4.0)
        self.assertGreaterEqual(centralised["nis_frac_all"], 0.90)
        self.assertGreater(centralised["joint_updates"], 0)


if __name__ == "__main__":
    unittest.main()
