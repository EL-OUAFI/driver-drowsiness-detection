from __future__ import annotations

import unittest

from driver_drowsiness.features.metrics import apply_displacements, eye_aspect_ratio, mouth_aspect_ratio


class MetricsTests(unittest.TestCase):
    def test_eye_aspect_ratio_for_open_eye(self) -> None:
        eye = [(0, 0), (1, 2), (3, 2), (4, 0), (3, -2), (1, -2)]
        self.assertEqual(eye_aspect_ratio(eye), 1.0)

    def test_mouth_aspect_ratio_for_open_mouth(self) -> None:
        mouth = [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 1),
            (4, 2),
            (5, 1),
            (6, 0),
            (5, -1),
            (4, -2),
            (3, -1),
            (2, -2),
            (1, -1),
        ]
        self.assertEqual(mouth_aspect_ratio(mouth), 0.6666666666666666)

    def test_apply_displacements(self) -> None:
        displaced = apply_displacements([(1.0, 1.0), (2.0, 2.0)], [(0.5, -0.5), (1.0, 1.0)])
        self.assertEqual(displaced, [(1.5, 0.5), (3.0, 3.0)])


if __name__ == "__main__":
    unittest.main()
