
# Driver Drowsiness Detection

Real-time driver drowsiness detection on the UTA-RLDD dataset.

This repository implements and compares two computer vision pipelines for binary **alert vs drowsy** classification from driver face frames:

- a landmark-based optical-flow pipeline
- a HOG-SVM pipeline combining HOG descriptors with EAR and MAR features

The strongest model is the **HOG-SVM** pipeline, which reached **0.84 precision**, **0.81 recall**, and **0.83 F1-score** on the **drowsy** class.

## Overview

Driver drowsiness detection is a safety-critical computer vision task. The goal is to detect visual signs such as prolonged eye closure, yawning, and reduced facial activity early enough to trigger an alert before driving performance degrades.

This repository focuses on:

- comparing two lightweight real-time detection strategies on the same dataset
- keeping feature extraction and modeling explicit and interpretable
- providing a clean, reusable Python codebase for training and evaluation

## Implemented Pipelines

### 1. Optical-Flow Pipeline

A landmark-based pipeline that analyzes facial motion across consecutive frames.

Main components:

- facial landmark detection with dlib's 68-point predictor
- optical-flow estimation between consecutive frames
- motion analysis on eye and mouth regions
- temporal reasoning through EAR and MAR dynamics

Implemented variants:

- Lucas-Kanade
- Lucas-Kanade Pyramid
- Horn-Schunck

### 2. HOG-SVM Pipeline

The best-performing pipeline in this repository. It combines appearance and geometric features for binary classification.

Main components:

- facial landmark detection
- extraction of eye and mouth regions of interest
- HOG feature extraction
- Eye Aspect Ratio (EAR)
- Mouth Aspect Ratio (MAR)
- `StandardScaler + SVC` classification pipeline

## Results

### HOG-SVM

- **Precision:** 0.84
- **Recall:** 0.81
- **F1-score:** 0.83
- **Role:** best-performing model

### Optical Flow

| Method | Accuracy | Precision | Recall |
| --- | --- | --- | --- |
| Lucas-Kanade | 0.75 | 0.77 | 0.70 |
| Lucas-Kanade Pyramid | 0.625 | 0.69 | 0.45 |
| Horn-Schunck | 0.71 | 0.76 | 0.62 |

Among the optical-flow variants, **Lucas-Kanade** achieved the strongest results.

## Repository Structure

```text
.
├── docs/
├── src/driver_drowsiness/
│   ├── cli.py
│   ├── data.py
│   ├── evaluation.py
│   ├── features/
│   └── pipelines/
├── tests/
├── pyproject.toml
└── README.md


Important modules:

* `src/driver_drowsiness/pipelines/hog_svm.py`: end-to-end training pipeline for the strongest model
* `src/driver_drowsiness/pipelines/optical_flow.py`: optical-flow-based pipeline for sequence evaluation
* `src/driver_drowsiness/features/`: landmark processing, HOG extraction, EAR, and MAR computation
* `src/driver_drowsiness/data.py`: dataset discovery, sequence grouping, and manifest generation

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Dependencies

This project requires:

* the UTA-RLDD dataset
* `dlib`
* `shape_predictor_68_face_landmarks.dat`

The dataset and pretrained landmark model are not included in this repository.

## Expected Dataset Layout

```text
dataset/
├── alert/
│   ├── 0001_01.jpg
│   ├── 0001_02.jpg
│   └── ...
└── tired/
    ├── 0137_01.jpg
    ├── 0137_02.jpg
    └── ...
```

The loader normalizes `tired` to the label `drowsy`.

## Usage

Build a dataset manifest:

```bash
drowsiness build-manifest --dataset-root /path/to/UTA-RLDD --output data/uta_rldd_manifest.csv
```

Train the HOG-SVM model:

```bash
drowsiness train-hog-svm \
  --dataset-root /path/to/UTA-RLDD \
  --predictor /path/to/shape_predictor_68_face_landmarks.dat \
  --artifacts-dir outputs/hog_svm
```

Evaluate the optical-flow pipeline:

```bash
drowsiness evaluate-optical-flow \
  --dataset-root /path/to/UTA-RLDD \
  --predictor /path/to/shape_predictor_68_face_landmarks.dat \
  --max-sequences 50 \
  --output outputs/optical_flow/report.json
```

## Notes

* the repository separates data indexing, feature extraction, modeling, and evaluation
* the command-line interface is designed so the workflow can be rerun without editing source files
* `src/` contains the reusable implementation
* `tests/` contains validation for dependency-light logic

```
```
