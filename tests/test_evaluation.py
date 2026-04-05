from __future__ import annotations

import unittest

from driver_drowsiness.evaluation import binary_classification_report


class EvaluationTests(unittest.TestCase):
    def test_binary_classification_report(self) -> None:
        report = binary_classification_report(
            ["alert", "drowsy", "drowsy", "alert"],
            ["alert", "drowsy", "alert", "alert"],
        )

        self.assertEqual(report["accuracy"], 0.75)
        self.assertEqual(report["precision"], 1.0)
        self.assertEqual(report["recall"], 0.5)
        self.assertEqual(report["f1"], 0.6667)
        self.assertEqual(report["confusion_matrix"], {"tp": 1, "tn": 2, "fp": 0, "fn": 1})


if __name__ == "__main__":
    unittest.main()
