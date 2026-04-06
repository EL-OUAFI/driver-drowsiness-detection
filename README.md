# Driver Drowsiness Detection

Real-time driver drowsiness detection on the UTA-RLDD dataset, developed at CentraleSupelec between January 2024 and April 2024.

This project compares two computer-vision pipelines for binary alert vs drowsy classification from driver face frames:

- an optical-flow baseline driven by facial landmarks
- a HOG-SVM pipeline combining HOG descriptors with EAR and MAR features

The strongest model is the HOG-SVM pipeline, which reached `0.84` precision, `0.81` recall, and `0.83` F1-score on the `drowsy` class.

## Project Overview

Driver drowsiness detection is a safety-critical vision task: the objective is to identify signs such as prolonged eye closure and yawning early enough to trigger an alert before driving performance degrades.

This repository focuses on three things:

- comparing two different real-time detection strategies on the same dataset
- keeping the feature engineering and modeling steps explicit and interpretable
- packaging the project as a clean Python codebase instead of a one-off experiment

## Methods

### 1. Optical-flow baseline

The baseline tracks facial motion over time from consecutive frames. Dense optical flow is combined with facial landmarks to monitor eye and mouth dynamics through geometric signals related to drowsiness.

Main ideas:

- detect facial landmarks with dlib's 68-point predictor
- estimate inter-frame motion with Lucas-Kanade or Horn-Schunck optical flow
- project motion onto eye and mouth landmarks
- infer drowsiness from temporal EAR and MAR behavior

### 2. HOG-SVM pipeline

The best-performing pipeline extracts appearance and geometric features from the eyes and mouth, then trains an SVM classifier.

Main ideas:

- detect the face and facial landmarks
- crop eye and mouth regions of interest
- extract HOG descriptors from these regions
- append EAR and MAR as compact geometric features
- train a `StandardScaler + SVC` pipeline with grid search

## Results

| Pipeline | Precision | Recall | F1-score | Role |
| --- | --- | --- | --- | --- |
| HOG-SVM + EAR/MAR | 0.84 | 0.81 | 0.83 | Best-performing model |
| Optical-flow baseline | - | - | - | Real-time comparison baseline |

## Code Structure

```text
.
|-- src/driver_drowsiness/
|   |-- cli.py
|   |-- data.py
|   |-- evaluation.py
|   |-- features/
|   `-- pipelines/
|-- tests/
|-- pyproject.toml
`-- README.md
```

Important modules:

- `src/driver_drowsiness/pipelines/hog_svm.py`: end-to-end training pipeline for the strongest model
- `src/driver_drowsiness/pipelines/optical_flow.py`: optical-flow-based baseline for sequence evaluation
- `src/driver_drowsiness/features/`: landmark processing, HOG extraction, EAR, and MAR computation
- `src/driver_drowsiness/data.py`: dataset discovery, sequence grouping, and manifest generation

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Validation

Run the lightweight unit tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -q
```

Required external assets:

- the UTA-RLDD dataset
- `shape_predictor_68_face_landmarks.dat`

The dataset and landmark model are not committed to this repository.

## Expected Dataset Layout

```text
dataset/
|-- alert/
|   |-- 0001_01.jpg
|   |-- 0001_02.jpg
|   `-- ...
`-- tired/
    |-- 0137_01.jpg
    |-- 0137_02.jpg
    `-- ...
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

Evaluate the optical-flow baseline:

```bash
drowsiness evaluate-optical-flow \
  --dataset-root /path/to/UTA-RLDD \
  --predictor /path/to/shape_predictor_68_face_landmarks.dat \
  --max-sequences 50 \
  --output outputs/optical_flow/report.json
```

## Technical Notes

- the project separates data indexing, feature extraction, modeling, and evaluation
- heavy scientific dependencies are imported lazily so failure points stay explicit
- the command-line interface is designed so the full workflow can be rerun without editing source files
- the current repository emphasizes the final implementation rather than exploratory artifacts
