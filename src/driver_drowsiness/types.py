"""Shared data structures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class DrowsinessLabel(str, Enum):
    """Binary labels used across the repository."""

    ALERT = "alert"
    DROWSY = "drowsy"


@dataclass(frozen=True)
class FrameRecord:
    """Metadata for a single image frame."""

    path: Path
    label: DrowsinessLabel
    sequence_id: str
    frame_index: int


@dataclass(frozen=True)
class SequenceRecord:
    """Ordered frames belonging to the same logical recording."""

    sequence_id: str
    label: DrowsinessLabel
    frames: tuple[FrameRecord, ...]
