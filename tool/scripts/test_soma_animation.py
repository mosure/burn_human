import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from soma_animation import convert


class SomaInterchangeTests(unittest.TestCase):
    def test_matrix_units_identity_and_virtual_root(self):
        # Format fixture only, not a claim of SOMA body-model validation.
        names = ["Hips"] + [f"joint_{i}" for i in range(1, 77)]
        rig = dict(id="soma-77", joints=[dict(name=n) for n in names])
        matrices = Rotation.from_rotvec([[0, 0.3, 0]] * 77).as_matrix()
        poses = np.concatenate([np.eye(3)[None], matrices])[None]
        data = dict(joint_names=["Root"] + names, poses=poses, rotation_repr="matrix", keep_root=True,
                    transl=np.array([[100, 90, -200]]), unit="centimeters", identity_model_type="anny",
                    identity_coeffs=np.array([[0.25, 0.75]]), scale_params=np.array([[1.1]]),
                    absolute_pose=False, joint_orient=np.tile(np.eye(3), (78, 1, 1)))
        result = convert(data, rig, 20, "fixture")
        np.testing.assert_allclose(result["translations"], [[1, 0.9, -2]])
        np.testing.assert_allclose(result["poses"], np.tile([0, 0.3, 0], (1, 77, 1)), atol=1e-7)
        self.assertEqual(result["identity_coeffs"], [0.25, 0.75])
        self.assertEqual(len(result["joint_orient"]), 77)
        data["poses"][0, 4, 0, 0] = 9
        with self.assertRaises(ValueError):
            convert(data, rig, 20, "fixture")


if __name__ == "__main__":
    unittest.main()
