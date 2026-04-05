# Engineering Notes

## Starting Point

The project originally lived in two notebooks:

- an optical-flow notebook
- a HOG-SVM notebook

That is a very normal way to prototype a computer-vision project, but notebooks usually mix:

- experimentation
- file-system assumptions
- training code
- visualization
- one-off data cleaning

For a hiring manager or recruiter, that often makes it harder to evaluate engineering quality.

## Refactor Goals

This repository was restructured around a few practical goals:

1. make the core logic importable
2. isolate dataset assumptions
3. provide command-line entrypoints
4. separate the research baseline from the best-performing model
5. make future reruns easier with saved artifacts and manifest files

## Main Improvements Over The Notebook Version

- Colab-specific paths were removed from the implementation.
- Dataset discovery is handled centrally in `data.py`.
- The HOG-SVM pipeline now uses fixed-size ROIs so feature lengths stay stable by construction.
- Training uses a proper `StandardScaler + SVC` pipeline with grid search instead of repeated batch fitting.
- The optical-flow baseline is represented as a reusable class instead of a notebook loop.
- Lightweight tests cover geometry, indexing, and metrics logic without requiring the full scientific stack.

## Intentional Constraints

- The UTA-RLDD dataset is not committed.
- The dlib landmark predictor is not committed.
- End-to-end validation still needs to be run locally on a machine with the required dependencies installed.

Those constraints are common in CV projects and are documented explicitly rather than hidden.
