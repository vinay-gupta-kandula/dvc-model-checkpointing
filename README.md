# Model Checkpointing with DVC

This project contains two versions of a Random Forest model trained on the
breast cancer dataset. Git stores the code and DVC stores the model files.

## Setup

```bash
pip install -r requirements.txt
```

## Run training

```bash
python train.py
```

This creates `model.joblib` and records the test accuracy in `metrics.json`.

## Restore a previous model

The repository includes a local DVC remote in `dvc_storage/`. To restore a
checkpoint, check out its Git tag and pull the matching model:

```bash
git checkout v1.0
dvc pull
```

```bash
git checkout v2.0
dvc pull
```

The two model versions are kept separately so either checkpoint can be
recovered at any time.
