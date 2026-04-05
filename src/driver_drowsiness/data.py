"""Dataset indexing utilities for UTA-RLDD-style frame folders."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .exceptions import DatasetStructureError
from .types import DrowsinessLabel, FrameRecord, SequenceRecord

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}
LABEL_ALIASES = {
    "alert": DrowsinessLabel.ALERT,
    "awake": DrowsinessLabel.ALERT,
    "drowsy": DrowsinessLabel.DROWSY,
    "sleepy": DrowsinessLabel.DROWSY,
    "tired": DrowsinessLabel.DROWSY,
}


def infer_label_from_path(path: Path) -> DrowsinessLabel:
    """Infer the binary label from directory names or the file name."""
    candidates = [part.lower() for part in path.parts]
    candidates.append(path.stem.lower())

    for candidate in candidates:
        for token, label in LABEL_ALIASES.items():
            if token in candidate:
                return label

    raise DatasetStructureError(
        f"Could not infer a label from '{path}'. Expected a path containing "
        "'alert', 'awake', 'drowsy', 'sleepy', or 'tired'."
    )


def extract_sequence_id(path: Path) -> str:
    """Derive a stable sequence identifier from a frame path."""
    stem = path.stem
    token = re.split(r"[-_. ]+", stem)[0]
    if token:
        return token
    return path.parent.name or stem


def extract_frame_index(path: Path) -> int:
    """Extract a sortable frame index from the file name."""
    digits = re.findall(r"\d+", path.stem)
    if not digits:
        return 0
    return int(digits[-1])


def discover_frame_records(dataset_root: Path | str) -> list[FrameRecord]:
    """Recursively scan the dataset directory and return sorted frame records."""
    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    records: list[FrameRecord] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = infer_label_from_path(path)
        records.append(
            FrameRecord(
                path=path,
                label=label,
                sequence_id=extract_sequence_id(path),
                frame_index=extract_frame_index(path),
            )
        )

    if not records:
        raise DatasetStructureError(f"No image frames found under: {root}")

    return sorted(records, key=lambda item: (item.label.value, item.sequence_id, item.frame_index))


def group_sequences(records: Iterable[FrameRecord]) -> list[SequenceRecord]:
    """Group and order frames into sequences."""
    grouped: dict[tuple[DrowsinessLabel, str], list[FrameRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.label, record.sequence_id)].append(record)

    sequences: list[SequenceRecord] = []
    for (label, sequence_id), frames in grouped.items():
        ordered = tuple(sorted(frames, key=lambda item: item.frame_index))
        sequences.append(SequenceRecord(sequence_id=sequence_id, label=label, frames=ordered))

    return sorted(sequences, key=lambda item: (item.label.value, item.sequence_id))


def write_manifest(records: Iterable[FrameRecord], output_path: Path | str) -> Path:
    """Persist a flat CSV manifest to disk."""
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "label", "sequence_id", "frame_index"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "path": str(record.path),
                    "label": record.label.value,
                    "sequence_id": record.sequence_id,
                    "frame_index": record.frame_index,
                }
            )
    return output


def read_manifest(manifest_path: Path | str) -> list[FrameRecord]:
    """Load frame metadata from a CSV manifest."""
    manifest = Path(manifest_path).expanduser().resolve()
    with manifest.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            FrameRecord(
                path=Path(row["path"]),
                label=DrowsinessLabel(row["label"]),
                sequence_id=row["sequence_id"],
                frame_index=int(row["frame_index"]),
            )
            for row in reader
        ]
