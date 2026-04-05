"""Facial landmark detection using dlib's 68-point predictor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..dependencies import require_module

LEFT_EYE_INDEXES = tuple(range(42, 48))
RIGHT_EYE_INDEXES = tuple(range(36, 42))
OUTER_MOUTH_INDEXES = tuple(range(48, 60))


@dataclass(frozen=True)
class FaceLandmarks:
    """Structured access to the 68-point face landmarks."""

    all_points: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]

    def slice(self, indexes: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
        return tuple(self.all_points[index] for index in indexes)

    @property
    def left_eye(self) -> tuple[tuple[int, int], ...]:
        return self.slice(LEFT_EYE_INDEXES)

    @property
    def right_eye(self) -> tuple[tuple[int, int], ...]:
        return self.slice(RIGHT_EYE_INDEXES)

    @property
    def outer_mouth(self) -> tuple[tuple[int, int], ...]:
        return self.slice(OUTER_MOUTH_INDEXES)


class DlibLandmarkDetector:
    """Lazy wrapper around dlib's face detector and shape predictor."""

    def __init__(self, predictor_path: str | Path) -> None:
        self.predictor_path = Path(predictor_path).expanduser().resolve()
        self._detector = None
        self._predictor = None

    def _ensure_backend(self) -> None:
        dlib = require_module("dlib", "pip install -e .")
        if self._detector is None:
            self._detector = dlib.get_frontal_face_detector()
        if self._predictor is None:
            self._predictor = dlib.shape_predictor(str(self.predictor_path))

    def detect_first(self, frame) -> FaceLandmarks | None:
        """Return the largest detected face in a frame."""
        cv2 = require_module("cv2")
        self._ensure_backend()

        gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._detector(gray, 0)
        if not faces:
            return None

        face = max(faces, key=lambda rect: rect.width() * rect.height())
        shape = self._predictor(gray, face)
        points = tuple((shape.part(i).x, shape.part(i).y) for i in range(68))
        bbox = (face.left(), face.top(), face.right(), face.bottom())
        return FaceLandmarks(all_points=points, bbox=bbox)
