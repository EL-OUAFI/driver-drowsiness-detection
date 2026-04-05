"""HOG-based feature engineering for the strongest reported pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from ..dependencies import require_module
from .landmarks import FaceLandmarks
from .metrics import bounding_box, eye_aspect_ratio, mouth_aspect_ratio


@dataclass(frozen=True)
class HogFeatureConfig:
    """Configuration for fixed-size region descriptors."""

    eye_size: tuple[int, int] = (48, 32)
    mouth_size: tuple[int, int] = (64, 32)
    orientations: int = 8
    pixels_per_cell: tuple[int, int] = (8, 8)
    cells_per_block: tuple[int, int] = (2, 2)
    block_norm: str = "L2-Hys"
    padding: int = 4
    min_roi_size: int = 8


def _crop_region(gray_frame, points, padding: int):
    x1, y1, x2, y2 = bounding_box(points, padding=padding)
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, gray_frame.shape[1] - 1)
    y2 = min(y2, gray_frame.shape[0] - 1)
    return gray_frame[y1 : y2 + 1, x1 : x2 + 1]


def _descriptor(roi, size, config: HogFeatureConfig):
    cv2 = require_module("cv2")
    skfeature = require_module("skimage.feature")

    resized = cv2.resize(roi, size, interpolation=cv2.INTER_AREA)
    return skfeature.hog(
        resized,
        orientations=config.orientations,
        pixels_per_cell=config.pixels_per_cell,
        cells_per_block=config.cells_per_block,
        block_norm=config.block_norm,
        feature_vector=True,
    )


def extract_feature_vector(frame, landmarks: FaceLandmarks, config: HogFeatureConfig | None = None):
    """Extract HOG, EAR, and MAR features from a single frame."""
    cv2 = require_module("cv2")
    np = require_module("numpy")

    cfg = config or HogFeatureConfig()
    gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    left_eye_roi = _crop_region(gray, landmarks.left_eye, cfg.padding)
    right_eye_roi = _crop_region(gray, landmarks.right_eye, cfg.padding)
    mouth_roi = _crop_region(gray, landmarks.outer_mouth, cfg.padding)

    rois = [left_eye_roi, right_eye_roi, mouth_roi]
    if any(min(roi.shape[:2]) < cfg.min_roi_size for roi in rois):
        return None

    left_hog = _descriptor(left_eye_roi, cfg.eye_size, cfg)
    right_hog = _descriptor(right_eye_roi, cfg.eye_size, cfg)
    mouth_hog = _descriptor(mouth_roi, cfg.mouth_size, cfg)

    ear = min(eye_aspect_ratio(landmarks.left_eye), eye_aspect_ratio(landmarks.right_eye))
    mar = mouth_aspect_ratio(landmarks.outer_mouth)
    return np.concatenate((left_hog, right_hog, mouth_hog, np.asarray([ear, mar], dtype=float)))
