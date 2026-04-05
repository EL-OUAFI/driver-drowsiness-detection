"""Optical-flow baseline for sequence-level drowsiness detection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..data import discover_frame_records, group_sequences
from ..dependencies import require_module
from ..evaluation import binary_classification_report
from ..features.landmarks import DlibLandmarkDetector, FaceLandmarks
from ..features.metrics import apply_displacements, eye_aspect_ratio, mouth_aspect_ratio
from ..types import DrowsinessLabel, SequenceRecord

MASK_X = [[-1, 1], [-1, 1]]
MASK_Y = [[-1, -1], [1, 1]]
MASK_T1 = [[-1, -1], [-1, -1]]
MASK_T2 = [[1, 1], [1, 1]]
MASK_LAPLACIAN = [
    [1 / 12, 1 / 6, 1 / 12],
    [1 / 6, 0, 1 / 6],
    [1 / 12, 1 / 6, 1 / 12],
]


@dataclass(frozen=True)
class OpticalFlowConfig:
    """Thresholds and algorithm settings for the baseline."""

    method: str = "lucas_kanade"
    ear_threshold: float = 0.24
    mar_threshold: float = 0.65
    consecutive_frames: int = 2
    landmark_refresh_interval: int = 5
    horn_schunck_alpha: float = 15.0
    horn_schunck_iterations: int = 300


def _compute_gradients(previous_frame, current_frame):
    signal = require_module("scipy.signal")
    np = require_module("numpy")
    fx = signal.convolve2d(previous_frame, np.asarray(MASK_X), mode="same") + signal.convolve2d(
        current_frame,
        np.asarray(MASK_X),
        mode="same",
    )
    fy = signal.convolve2d(previous_frame, np.asarray(MASK_Y), mode="same") + signal.convolve2d(
        current_frame,
        np.asarray(MASK_Y),
        mode="same",
    )
    ft = signal.convolve2d(previous_frame, np.asarray(MASK_T1), mode="same") + signal.convolve2d(
        current_frame,
        np.asarray(MASK_T2),
        mode="same",
    )
    return fx, fy, ft


def lucas_kanade_dense(fx, fy, ft, frame_shape, window_size: int = 3):
    """Estimate dense flow with a local least-squares Lucas-Kanade solver."""
    np = require_module("numpy")

    height, width = frame_shape
    u = np.zeros((height, width))
    v = np.zeros((height, width))
    radius = window_size // 2

    for y in range(radius, height - radius):
        for x in range(radius, width - radius):
            patch_fx = fx[y - radius : y + radius + 1, x - radius : x + radius + 1].reshape(-1)
            patch_fy = fy[y - radius : y + radius + 1, x - radius : x + radius + 1].reshape(-1)
            patch_ft = ft[y - radius : y + radius + 1, x - radius : x + radius + 1].reshape(-1)
            a = np.vstack((patch_fx, patch_fy)).T
            b = -patch_ft
            ata = a.T @ a
            if np.linalg.det(ata) < 1e-6:
                continue
            result = np.linalg.pinv(ata) @ a.T @ b
            u[y, x], v[y, x] = result

    return np.dstack((u, v))


def horn_schunck_dense(fx, fy, ft, frame_shape, alpha: float, iterations: int):
    """Estimate dense flow with the Horn-Schunck method."""
    np = require_module("numpy")
    signal = require_module("scipy.signal")

    height, width = frame_shape
    u = np.zeros((height, width))
    v = np.zeros((height, width))
    laplacian = np.asarray(MASK_LAPLACIAN)

    for _ in range(iterations):
        u_avg = signal.convolve2d(u, laplacian, mode="same")
        v_avg = signal.convolve2d(v, laplacian, mode="same")
        numerator = fx * u_avg + fy * v_avg + ft
        denominator = alpha**2 + fx**2 + fy**2
        u = u_avg - fx * (numerator / denominator)
        v = v_avg - fy * (numerator / denominator)

    return np.dstack((u, v))


def _sample_offsets(flow, points):
    offsets = []
    height, width = flow.shape[:2]
    for x, y in points:
        ix = min(max(int(round(x)), 0), width - 1)
        iy = min(max(int(round(y)), 0), height - 1)
        dx, dy = flow[iy, ix]
        offsets.append((float(dx), float(dy)))
    return offsets


def _transport_landmarks(landmarks: FaceLandmarks, flow) -> FaceLandmarks:
    moved_points = tuple(
        apply_displacements(landmarks.all_points, _sample_offsets(flow, landmarks.all_points))
    )
    return FaceLandmarks(all_points=moved_points, bbox=landmarks.bbox)


class OpticalFlowBaseline:
    """Sequence-level baseline that combines dense flow with EAR and MAR thresholds."""

    def __init__(self, predictor_path: str | Path, config: OpticalFlowConfig | None = None) -> None:
        self.config = config or OpticalFlowConfig()
        self.landmark_detector = DlibLandmarkDetector(predictor_path)

    def _dense_flow(self, previous_frame, current_frame):
        fx, fy, ft = _compute_gradients(previous_frame, current_frame)
        if self.config.method == "horn_schunck":
            return horn_schunck_dense(
                fx,
                fy,
                ft,
                previous_frame.shape,
                alpha=self.config.horn_schunck_alpha,
                iterations=self.config.horn_schunck_iterations,
            )
        return lucas_kanade_dense(fx, fy, ft, previous_frame.shape)

    def predict_sequence(self, sequence: SequenceRecord) -> DrowsinessLabel | None:
        """Predict the binary state for a whole ordered frame sequence."""
        cv2 = require_module("cv2")

        if len(sequence.frames) < 2:
            return None

        current_landmarks = None
        previous_gray = None
        drowsy_streak = 0

        for index, frame_record in enumerate(sequence.frames):
            frame = cv2.imread(str(frame_record.path))
            if frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if current_landmarks is None:
                current_landmarks = self.landmark_detector.detect_first(frame)
                previous_gray = gray
                if current_landmarks is None:
                    continue
                continue

            if index % self.config.landmark_refresh_interval == 0:
                refreshed_landmarks = self.landmark_detector.detect_first(frame)
                if refreshed_landmarks is not None:
                    current_landmarks = refreshed_landmarks
                    previous_gray = gray
                    continue

            flow = self._dense_flow(previous_gray, gray)
            current_landmarks = _transport_landmarks(current_landmarks, flow)
            left_eye = current_landmarks.left_eye
            right_eye = current_landmarks.right_eye
            mouth = current_landmarks.outer_mouth

            ear = min(eye_aspect_ratio(left_eye), eye_aspect_ratio(right_eye))
            mar = mouth_aspect_ratio(mouth)
            if ear < self.config.ear_threshold or mar > self.config.mar_threshold:
                drowsy_streak += 1
            else:
                drowsy_streak = 0

            previous_gray = gray
            if drowsy_streak >= self.config.consecutive_frames:
                return DrowsinessLabel.DROWSY

        return DrowsinessLabel.ALERT

    def evaluate(self, dataset_root: str | Path, *, max_sequences: int | None = None) -> dict[str, object]:
        """Evaluate the optical-flow baseline on discovered sequences."""
        records = discover_frame_records(dataset_root)
        sequences = group_sequences(records)
        if max_sequences is not None:
            sequences = sequences[:max_sequences]

        y_true: list[str] = []
        y_pred: list[str] = []
        skipped = 0

        for sequence in sequences:
            prediction = self.predict_sequence(sequence)
            if prediction is None:
                skipped += 1
                continue
            y_true.append(sequence.label.value)
            y_pred.append(prediction.value)

        if not y_true:
            raise ValueError("No evaluable sequences were found.")

        return {
            "config": self.config.__dict__,
            "sequences_evaluated": len(y_true),
            "sequences_skipped": skipped,
            "metrics": binary_classification_report(y_true, y_pred),
        }

    def evaluate_to_json(
        self,
        dataset_root: str | Path,
        output_path: str | Path,
        *,
        max_sequences: int | None = None,
    ) -> Path:
        """Run the evaluation and save the summary as JSON."""
        report = self.evaluate(dataset_root, max_sequences=max_sequences)
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        return output
