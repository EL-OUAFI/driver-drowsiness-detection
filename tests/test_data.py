from __future__ import annotations

import unittest
from pathlib import Path

from driver_drowsiness.data import (
    discover_frame_records,
    extract_frame_index,
    extract_sequence_id,
    group_sequences,
    infer_label_from_path,
)
from driver_drowsiness.types import DrowsinessLabel


class DataTests(unittest.TestCase):
    def test_infer_label_from_path_normalizes_tired(self) -> None:
        path = Path("/tmp/train/tired/0137_08.jpg")
        self.assertIs(infer_label_from_path(path), DrowsinessLabel.DROWSY)

    def test_extract_sequence_and_frame_index(self) -> None:
        path = Path("/tmp/train/alert/0001_12.jpg")
        self.assertEqual(extract_sequence_id(path), "0001")
        self.assertEqual(extract_frame_index(path), 12)

    def test_discover_records_and_group_sequences(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            alert_dir = tmp_path / "alert"
            tired_dir = tmp_path / "tired"
            alert_dir.mkdir()
            tired_dir.mkdir()

            for name in ("0001_01.jpg", "0001_02.jpg", "0002_01.jpg"):
                (alert_dir / name).write_bytes(b"test")
            for name in ("0137_01.jpg", "0137_02.jpg"):
                (tired_dir / name).write_bytes(b"test")

            records = discover_frame_records(tmp_path)
            sequences = group_sequences(records)

            self.assertEqual(len(records), 5)
            self.assertEqual([sequence.sequence_id for sequence in sequences], ["0001", "0002", "0137"])
            self.assertEqual(
                [sequence.label for sequence in sequences],
                [
                    DrowsinessLabel.ALERT,
                    DrowsinessLabel.ALERT,
                    DrowsinessLabel.DROWSY,
                ],
            )


if __name__ == "__main__":
    unittest.main()
