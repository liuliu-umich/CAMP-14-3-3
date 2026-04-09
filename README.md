# 14-3-3 Predictor

This repository contains a machine learning pipeline and a lightweight Flask web app for predicting 14-3-3 binding at serine/threonine phosphosites from protein sequence.

The current application supports three prediction modes:

- Single-site prediction for one user-specified `S/T` position
- Full-sequence scanning across all candidate `S/T` sites
- Sequence-level binding classification based on the highest-scoring site

## Overview

The project combines several feature sources for each candidate site, including:

- Sequence-context one-hot features
- ESM protein and residue embeddings
- IUPred2A disorder and ANCHOR2 scores
- DeePhase-derived sequence features
- Compactness / charge pattern features
- PhosphoLingo phosphorylation-related score

These features are passed into a trained classification model stored in [`model/1433model_20260223.pkl`](/Users/newuser/1433predictor/model/1433model_20260223.pkl).

## Main Files

- [`app.py`](/Users/newuser/1433predictor/app.py): Flask entry point for the web interface
- [`features.py`](/Users/newuser/1433predictor/features.py): feature extraction, validation, and prediction logic
- [`environment_torch.yml`](/Users/newuser/1433predictor/environment_torch.yml): Conda environment definition
- [`model/`](/Users/newuser/1433predictor/model): trained models and auxiliary data
- [`templates/`](/Users/newuser/1433predictor/templates): HTML pages for the web app
- [`static/style.css`](/Users/newuser/1433predictor/static/style.css): web UI styling
- `*.ipynb`: notebooks for data analysis, feature engineering, training, threshold tuning, and demos

## Requirements

This project is set up around:

- Python 3.9
- Conda
- PyTorch 1.13.1
- Flask 3.1.1
- Hugging Face `transformers`
- Biopython, scikit-learn, pandas, numpy, gensim, localCIDER, and related scientific packages

The included Conda environment file is the recommended way to reproduce the runtime environment.

## Setup

Create the environment:

```bash
conda env create -f environment_torch.yml
conda activate 1433predictor2026
```

## Important External Dependency

The code expects a PhosphoLingo checkpoint file that is not stored in this repository.

By default, [`features.py`](/Users/newuser/1433predictor/features.py) looks for:

```text
/Users/newuser/PhosphoLingo_ST_new.ckpt
```

You can override this by setting an environment variable before starting the app:

```bash
export PHOSPHOLINGO_MODEL_LOC=/absolute/path/to/your/PhosphoLingo_ST_new.ckpt
```

Optional environment variables:

```bash
export SEQUENCE_BINDING_THRESHOLD=0.5
export PHOSPHOLINGO_SITE_CHUNK_SIZE=16
```

## Run the Web App

Start the Flask app:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Web App Functions

### 1. Predict One Site

Input:

- A protein sequence
- A 1-based site index

Output:

- Predicted class
- Positive and negative probabilities
- Number of features used by the model

The selected residue must be `S` or `T`.

### 2. Score All S/T Sites

Input:

- A protein sequence

Output:

- A table of all candidate serine/threonine sites
- Per-site prediction
- Positive and negative probabilities

### 3. Predict Sequence Binding

Input:

- A protein sequence

Output:

- Sequence-level binding decision
- Best-scoring site
- Best-site residue
- Maximum probability across all candidate sites
- Threshold used for the final sequence-level call

## Input Rules

- Site positions are 1-based
- Single-site prediction only accepts positions containing `S` or `T`
- For sequence scanning, all `S/T` residues are evaluated automatically
- Non-standard amino acids are sanitized internally during feature generation

## Notes

- The ESM model is loaded from Hugging Face on first use, so the first prediction can be slower.
- BLAST `segmasker` is expected to be available in the environment or on `PATH`.
- Temporary files may be created during PhosphoLingo inference and removed automatically.
- Processing errors related to invalid temporary sequence formatting may be appended to `processing_errors.log`.

## Quick Test

The web page includes a simple example:

- Site: `8`
- Sequence: `ASAAAAAASAAAAAAT`

This is useful for confirming the app runs end-to-end.

## Development History

The notebooks in this repository document the broader workflow, including:

- exploratory analysis
- data augmentation
- dataset splitting
- feature extraction
- model training
- site-level threshold optimization
- sequence-level demo workflows

## License

No license file is currently included in this repository. Add one if you plan to distribute the project publicly.
