# 14-3-3 Predictor

This repository contains the code, trained models, notebooks, and web application materials associated with the manuscript currently under review:

`A Context-Aware 14-3-3 Binding Predictor Enabled by Data Augmentation, Multi-Scale Biological Features, and Protein Language Model Embeddings`

It provides a machine learning pipeline and a lightweight Flask web app for predicting 14-3-3 binding at serine/threonine phosphosites from protein sequence.

## Authors

`Liu Liu1,2*`, `Zhong Wang2`, `Xiaoqiang Huang3`

`1` Gilbert S. Omenn Department of Computational Medicine and Bioinformatics, Ann Arbor, MI 48109, USA  
`2` Department of Cardiac Surgery, Frankel Cardiovascular Center, The University of Michigan, Ann Arbor, MI 48109, USA  
`3` Center for Advanced Models for Translational Sciences and Therapeutics, Department of Internal Medicine, University of Michigan Medical School, 2800 Plymouth Road, Ann Arbor, MI 48109, USA  
`*` Correspondence: `luvul@umich.edu` (LL)

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

These features are passed into a trained classification model stored in [`model/1433model_20260223.pkl`](model/1433model_20260223.pkl).

## Main Files

- [`app.py`](app.py): Flask entry point for the web interface
- [`features.py`](features.py): feature extraction, validation, and prediction logic
- [`environment_torch.yml`](environment_torch.yml): Conda environment definition
- [`model/`](model): trained models and auxiliary data
- [`templates/`](templates): HTML pages for the web app
- [`static/style.css`](static/style.css): web UI styling
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

## Data and Model Availability

The repository includes:

- source code for feature extraction, model inference, and the Flask web interface
- trained 14-3-3 predictor model files in [`model/`](model)
- notebooks documenting data analysis, augmentation, feature engineering, training, and evaluation workflows
- project data and intermediate resources stored under [`data/`](data) and [`datasets/`](datasets)

The repository does not include:

- the external checkpoint used with the bundled `PhosphoLingo`-derived inference code for phosphorylation-related feature generation
- third-party model assets that may be downloaded separately at runtime, such as ESM weights from Hugging Face

These resources are not bundled here because they are maintained externally and may be subject to size, distribution, or third-party licensing constraints.

To run the full prediction pipeline, users should obtain the required checkpoint independently and point the code to its local path as described below.

## Third-Party Software and Licensing

This repository integrates or depends on several third-party resources for feature generation and model inference.

- `PhosphoLingo`: this repository includes `PhosphoLingo`-related source code under [`src/files/PhosphoLingo/`](src/files/PhosphoLingo), with local modifications for integration into this workflow. At the time of preparation, the original `PhosphoLingo` repository did not provide a clearly identified open-source license in its repository metadata. Users should review the original project directly and verify the applicable reuse and redistribution terms before reusing these components.
- `DeePhase`: this repository includes `DeePhase`-related files under [`src/files/DeePhase/`](src/files/DeePhase). The bundled `DeePhase` materials state that the work is for academic use only and is distributed under a `CC BY-NC 4.0` license. Reuse is therefore subject to attribution and non-commercial restrictions.
- `IUPred2A`: this repository includes `IUPred2A`-related files under [`src/files/iupred2a/`](src/files/iupred2a). The bundled license states that `IUPred2A` is available free of charge only to academic users, may not be used for commercial purposes, and may not be redistributed to others. Users should consult the original `IUPred2A` academic license before any reuse or redistribution.
- `IDRome (_2023_Tesei_IDRome)`: this project uses sequence-property calculations and/or pretrained SVR models derived from the `KULL-Centre/_2023_Tesei_IDRome` repository. This repository currently includes related model/data files such as [`model/svr_model_nu.joblib`](model/svr_model_nu.joblib), [`model/svr_model_SPR.joblib`](model/svr_model_SPR.joblib), and [`model/residues.csv`](model/residues.csv). The original repository is distributed under `GPL-3.0`. Any reuse or redistribution of code, models, or other derived components from that repository should follow the original `GPL-3.0` license terms.

Users are responsible for ensuring that their use of these third-party resources complies with the respective original licenses, terms of use, and citation requirements.

## Important External Dependency

The code expects a compatible `PhosphoLingo` checkpoint file that is not stored in this repository.

By default, [`features.py`](features.py) looks for:

```text
/path/to/PhosphoLingo_ST_new.ckpt
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

## Reference

If you use this repository, please cite:

```text
Liu L, Wang Z, Huang X. A Context-Aware 14-3-3 Binding Predictor Enabled by Data Augmentation, Multi-Scale Biological Features, and Protein Language Model Embeddings. Manuscript under review.
```

## License

This repository is released under the `MIT License`. See the [`LICENSE`](LICENSE) file for details.
