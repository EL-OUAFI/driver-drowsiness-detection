"""Training pipeline for the HOG-SVM drowsiness detector."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..data import discover_frame_records
from ..dependencies import require_module
from ..evaluation import binary_classification_report
from ..exceptions import DatasetStructureError
from ..features.hog import HogFeatureConfig, extract_feature_vector
from ..features.landmarks import DlibLandmarkDetector
from ..types import FrameRecord


@dataclass(frozen=True)
class HogSvmTrainingConfig:
    """Hyperparameters for the HOG-SVM experiment."""

    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    class_weight: str = "balanced"
    c_values: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    gamma_values: tuple[float, ...] = (1.0, 0.1, 0.01, 0.001)
    kernels: tuple[str, ...] = ("rbf",)


@dataclass(frozen=True)
class TrainingSummary:
    """Serializable summary of a training run."""

    samples_used: int
    samples_skipped: int
    best_params: dict[str, object]
    metrics: dict[str, object]
    model_path: str
    report_path: str
    feature_config: dict[str, object]
    training_config: dict[str, object]


class HogSvmTrainer:
    """End-to-end training for the landmark + HOG + SVM pipeline."""

    def __init__(
        self,
        predictor_path: str | Path,
        *,
        feature_config: HogFeatureConfig | None = None,
        training_config: HogSvmTrainingConfig | None = None,
    ) -> None:
        self.landmark_detector = DlibLandmarkDetector(predictor_path)
        self.feature_config = feature_config or HogFeatureConfig()
        self.training_config = training_config or HogSvmTrainingConfig()

    def build_dataset(self, records: list[FrameRecord]):
        """Load frames and convert them to feature vectors."""
        cv2 = require_module("cv2")
        np = require_module("numpy")

        features = []
        labels: list[str] = []
        skipped: list[str] = []

        for record in records:
            frame = cv2.imread(str(record.path))
            if frame is None:
                skipped.append(str(record.path))
                continue

            landmarks = self.landmark_detector.detect_first(frame)
            if landmarks is None:
                skipped.append(str(record.path))
                continue

            vector = extract_feature_vector(frame, landmarks, self.feature_config)
            if vector is None:
                skipped.append(str(record.path))
                continue

            features.append(vector)
            labels.append(record.label.value)

        if not features:
            raise DatasetStructureError("No usable samples were extracted from the dataset.")

        return np.vstack(features), np.asarray(labels), skipped

    def train(self, records: list[FrameRecord], artifacts_dir: str | Path) -> TrainingSummary:
        """Train, evaluate, and persist the best HOG-SVM model."""
        model_selection = require_module("sklearn.model_selection")
        pipeline_mod = require_module("sklearn.pipeline")
        preprocessing_mod = require_module("sklearn.preprocessing")
        svm_mod = require_module("sklearn.svm")

        features, labels, skipped = self.build_dataset(records)
        unique_labels = sorted(set(labels.tolist()))
        if len(unique_labels) < 2:
            raise DatasetStructureError(
                "Training requires both alert and drowsy samples after filtering."
            )

        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            features,
            labels,
            test_size=self.training_config.test_size,
            random_state=self.training_config.random_state,
            stratify=labels,
        )

        class_counts = {}
        for label in unique_labels:
            class_counts[label] = int((y_train == label).sum())
        min_class_count = min(class_counts.values())
        if min_class_count < 2:
            raise DatasetStructureError("Need at least two samples per class for cross-validation.")

        estimator = pipeline_mod.Pipeline(
            [
                ("scaler", preprocessing_mod.StandardScaler()),
                ("svc", svm_mod.SVC(class_weight=self.training_config.class_weight)),
            ]
        )

        search = model_selection.GridSearchCV(
            estimator=estimator,
            param_grid={
                "svc__C": list(self.training_config.c_values),
                "svc__gamma": list(self.training_config.gamma_values),
                "svc__kernel": list(self.training_config.kernels),
            },
            cv=min(self.training_config.cv_folds, min_class_count),
            scoring="f1",
            n_jobs=-1,
            refit=True,
        )
        search.fit(x_train, y_train)

        predictions = search.predict(x_test).tolist()
        metrics = binary_classification_report(y_test.tolist(), predictions)

        output_dir = Path(artifacts_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / "hog_svm_model.pkl"
        report_path = output_dir / "training_report.json"

        with model_path.open("wb") as handle:
            pickle.dump(
                {
                    "estimator": search.best_estimator_,
                    "best_params": search.best_params_,
                    "feature_config": asdict(self.feature_config),
                    "training_config": asdict(self.training_config),
                },
                handle,
            )

        summary = TrainingSummary(
            samples_used=int(features.shape[0]),
            samples_skipped=len(skipped),
            best_params=search.best_params_,
            metrics=metrics,
            model_path=str(model_path),
            report_path=str(report_path),
            feature_config=asdict(self.feature_config),
            training_config=asdict(self.training_config),
        )

        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(summary), handle, indent=2, default=str)

        return summary

    def train_from_dataset(self, dataset_root: str | Path, artifacts_dir: str | Path) -> TrainingSummary:
        """Discover records from disk, then train the pipeline."""
        records = discover_frame_records(dataset_root)
        return self.train(records, artifacts_dir)
