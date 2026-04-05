# Driver Drowsiness Detection

This repository turns an academic computer-vision project into a reproducible Python codebase. It implements and documents two real-time drowsiness-detection pipelines inspired by work done at CentraleSupelec between January 2024 and April 2024 on the UTA-RLDD dataset:

- a landmark-driven optical-flow baseline
- a HOG-SVM pipeline combining HOG descriptors with EAR and MAR

The original notebooks are kept in [`notebooks/`](notebooks) for traceability, while the recruiter-facing implementation lives in `src/`.

## Why This Repository Is Structured This Way

The original project existed as two exploratory notebooks. That format is good for experimentation, but it is not ideal for:

- reproducibility
- code review
- testing
- packaging
- command-line execution
- communicating engineering maturity

This refactor keeps the research intent but packages it like production-quality applied ML work.

## Highlights

- `src/driver_drowsiness/`: reusable Python package with typed modules and a CLI
- `src/driver_drowsiness/pipelines/hog_svm.py`: training pipeline for the strongest reported model
- `src/driver_drowsiness/pipelines/optical_flow.py`: cleaned optical-flow baseline
- `src/driver_drowsiness/data.py`: dataset discovery and manifest generation for UTA-RLDD-style frame folders
- `tests/`: unit tests for dependency-free geometry, metrics, and dataset-indexing logic
- `docs/engineering-notes.md`: design decisions and how the notebook prototype was professionalized

## Repository Layout

```text
.
|-- docs/
|-- notebooks/
|-- src/driver_drowsiness/
|   |-- cli.py
|   |-- data.py
|   |-- evaluation.py
|   |-- features/
|   |-- pipelines/
|   `-- ...
|-- tests/
|-- pyproject.toml
`-- README.md
```

## Installation

Create a virtual environment, then install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### Notes on Dependencies

- `dlib` is required because the original project uses the 68-point facial landmark predictor.
- You will also need the `shape_predictor_68_face_landmarks.dat` file from dlib's public model zoo.
- The dataset itself is not committed here because of size and licensing constraints.

## Expected Dataset Layout

The indexing code is intentionally flexible, but it works best with a tree that contains `alert`, `drowsy`, or `tired` in directory names:

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

`tired` is normalized to the label `drowsy` so the code can stay consistent.

## Command Line Usage

Build a CSV manifest of the dataset:

```bash
drowsiness build-manifest --dataset-root /path/to/UTA-RLDD --output data/uta_rldd_manifest.csv
```

Train the HOG-SVM pipeline:

```bash
drowsiness train-hog-svm \
  --dataset-root /path/to/UTA-RLDD \
  --predictor /path/to/shape_predictor_68_face_landmarks.dat \
  --artifacts-dir outputs/hog_svm
```

Evaluate the optical-flow baseline on a subset of sequences:

```bash
drowsiness evaluate-optical-flow \
  --dataset-root /path/to/UTA-RLDD \
  --predictor /path/to/shape_predictor_68_face_landmarks.dat \
  --max-sequences 50 \
  --output outputs/optical_flow/report.json
```

## Reported Results

From the original study summary:

- best model: HOG-SVM
- precision: `0.84`
- recall: `0.81`
- F1-score: `0.83`

These numbers should be treated as the project report benchmark. Reproducing them in a fresh environment requires the original dataset split, landmark model, and the same preprocessing assumptions.

## What A Reviewer Should Notice

- the repository preserves the original research artifact but no longer depends on notebooks to run
- the code separates data indexing, feature extraction, modeling, and evaluation concerns
- heavy dependencies are imported lazily where possible so failure modes are explicit
- the CLI is designed to be usable by someone who did not author the project

## Next Steps

Before publishing publicly, you should still:

1. add one or two saved benchmark reports under `docs/` or `outputs/examples/`
2. run the code on your local machine with the real dataset and landmark file
3. update the README with a short "reproducibility status" note once you have rerun the final experiments
