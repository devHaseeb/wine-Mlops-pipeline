# Wine MLOps Pipeline

## Project Overview

This project implements an end-to-end MLOps pipeline for Wine cultivar classification using the scikit-learn Wine dataset.

The project demonstrates:

* Data loading and validation
* Stratified train-test splitting
* Random Forest and Gradient Boosting models
* Hyperparameter experimentation
* 5-fold stratified cross-validation
* MLflow experiment tracking
* MLflow Model Registry
* Champion model selection
* Automated model evaluation
* Unit testing
* MLOps quality gates
* GitHub Actions CI/CD
* Git feature branching and merge conflict resolution

## Project Structure

```text
wine-mlops-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── train.py
│   └── evaluate.py
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   └── test_model_gate.py
├── .gitignore
├── Makefile
├── README.md
└── requirements.txt
```

## Installation

Create and activate a Python 3.10 virtual environment:

```bash
python -m venv venv
source venv/Scripts/activate
```

Install dependencies:

```bash
make install
```

## Makefile Commands

Run linting:

```bash
make lint
```

Run tests:

```bash
make test
```

Train the models and track experiments:

```bash
make train
```

Clean Python cache files:

```bash
make clean
```

## Model Training

The training pipeline evaluates:

* 3 Random Forest configurations
* 3 Gradient Boosting configurations
* 5-fold stratified cross-validation

The models are evaluated using:

* Accuracy
* Macro F1-score
* Log Loss

The best configuration is selected using validation Macro F1-score.

## MLflow

The MLflow experiment is:

```text
Wine-Cultivar-Classification
```

The selected model is registered as:

```text
WineClassifier
```

The champion model is identified using the:

```text
champion
```

alias.

## Model Evaluation

The registered champion model is evaluated on the held-out test set using:

* Test Accuracy
* Test Macro F1
* Test Log Loss

The champion model achieved:

```text
Test Accuracy: 1.0000
Test Macro F1: 1.0000
Test Log Loss: 0.1059
```

## Automated Quality Gates

The project includes three model quality gates:

1. Validation Macro F1 must be at least 0.88.
2. Batch inference latency must be no more than 30 ms.
3. Model predictions must contain only valid class indices: 0, 1, or 2.

## CI/CD

GitHub Actions automatically runs on:

* Push to `main`
* Pull requests targeting `main`

The CI workflow performs:

```text
make install
make lint
make test
```

The test suite includes data validation and model quality-gate tests.

## Git Collaboration

The project uses a feature-branch workflow.

Example feature branch:

```text
conflict-simulation
```

A merge conflict was intentionally created by modifying the same CI configuration line differently on `main` and the feature branch. The conflict was manually resolved and the resulting merge was committed and pushed to `main`.

## Results

The best model configuration achieved a validation Macro F1-score of:

```text
0.9789
```

The final champion model was:

```text
RandomForest_Config_1
```
