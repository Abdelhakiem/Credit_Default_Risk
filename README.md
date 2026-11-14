# Credit Default Risk

A CatBoost pipeline that estimates the probability of default (`TARGET = 1`) for Credit applications. The repository covers data preparation, feature engineering, final model training, MLflow registration, evaluation, and single or batch inference from engineered features.

Contents

- [Quick start](#quick-start)
- [Predict from a saved model](#predict-from-a-saved-model)
- [Configuration](#configuration)
- [ML system and stages](#ml-system-and-stages)
- [Project structure](#project-structure)
- [Development checks](#development-checks)

## Quick start

Run these commands from the repository root. Install [uv](https://docs.astral.sh/uv/) and use Python 3.12 (`.python-version` records the project version).

```bash
uv sync --extra dev
```

Prepare the stratified train/test split, then engineer features:

```bash
uv run python -m credit_default_risk.data
uv run python -m credit_default_risk.features
```

The first command writes `data/interim/train.parquet` and `test.parquet`. The second writes their feature-engineered counterparts to `data/processed/`. The default split is 85% train and 15% test with seed 42. Both commands emit JSON logs.

Training needs a running MLflow server because the existing experiment uses `mlflow-artifacts:/` for artifact storage. In a separate terminal, from the repository root, start the server against the project database and artifact directory:

```bash
uv run mlflow server \
  --backend-store-uri "sqlite:///$(pwd)/mlflow.db" \
  --artifacts-destination "$(pwd)/mlartifacts" \
  --host 127.0.0.1 --port 5000
```

Then run training in the first terminal:

```bash
uv run python -m credit_default_risk.train
```

`train.py` reads the processed splits, selects the 60 final features, and reconstructs CatBoost parameters from the configured winning MLflow run. If that run is unavailable or invalid, it logs a warning and uses the configured default parameters. It evaluates on the held-out test set, writes `model.cbm`, `model.pkl`, `manifest.json`, `evaluation.json`, and `registration.json` under `artifacts/catboost_final/`, and registers a new version of `credit-default-risk-catboost` in MLflow. Training does **not** move the `champion` alias; promotion is a separate decision. The current training manifest has no review threshold, so inference returns a probability without a review decision.

## Predict from a saved model

Prediction expects **one object with all 60 engineered features**, or a JSON list of such objects. It does not accept raw application and history tables. Feature names come from the saved `manifest.json`; categorical values can be strings or `null`, and numeric values can be numbers or `null`.

To make a local example from the first processed test row:

```bash
uv run python - <<'PY'
import json
import pandas as pd
from credit_default_risk.config import PROCESSED_TEST_PATH
from credit_default_risk.predict import Predictor

predictor = Predictor.load(source="local")
columns = predictor.manifest["feature_columns"]
row = pd.read_parquet(PROCESSED_TEST_PATH, columns=columns).head(1)
with open("example-input.json", "w", encoding="utf-8") as output:
    json.dump(json.loads(row.to_json(orient="records"))[0], output, indent=2)
PY
```

Score it with the local bundle:

```bash
uv run python -m credit_default_risk.predict --input example-input.json --source local
```

Omit `--source local` to try the MLflow `champion` alias first. If registry loading fails, the predictor logs a warning and uses the local bundle. `--artifacts PATH` selects a different local bundle directory. Output is JSON containing `default_probability`, `review_flag`, `decision_threshold`, and `model_version`. `review_flag` and `decision_threshold` are `null` when the manifest has no threshold. For a batch, make the input file a JSON list; output will be a list in the same order. Avoid committing applicant feature files.

## Configuration

`src/credit_default_risk/config.py` uses `pydantic-settings`. Environment variables override its defaults when Python starts. Useful variables are `RAW_DIR`, `INTERIM_DIR`, `PROCESSED_DIR`, `ARTIFACT_DIR`, `MLFLOW_TRACKING_URI`, `MODEL_NAME`, `MODEL_ALIAS`, `WINNING_RUN_ID`, `TEST_SIZE`, `RANDOM_STATE`, `PROPERTY_SUFFIX`, and `LOG_LEVEL`. `PROJECT_ROOT` can relocate the default directories; individual directory variables can override them separately. Set `MLFLOW_TRACKING_URI` if the server is not at `http://127.0.0.1:5000`.

For example:

```bash
ARTIFACT_DIR=/path/to/model MLFLOW_TRACKING_URI=http://localhost:5000 \
  uv run python -m credit_default_risk.predict --input example-input.json
```

## ML system and stages

```text
Raw application + history CSVs
            │ data.py: clean, aggregate history, stratified split
            ▼
      data/interim/*.parquet
            │ features.py: time/ratio features, MEDI property group
            ▼
     data/processed/*.parquet
            │ train.py: fit CatBoost, evaluate, save and register
            ▼
  local model bundle + MLflow model version
            │ predict.py: load registry alias or local bundle
            ▼
     default probability per record
```

The model consumes 60 ordered features recorded in its manifest. `SK_ID_CURR` and `TARGET` are kept in training tables but excluded from model inputs. The final test split is used for reporting, not tuning. The notebooks contain exploration, selection, and calibration research. FastAPI serving and Docker packaging are planned deployment stages; `api.py` is currently a placeholder.

## Project structure

| Path                                               | Purpose                                                            |
| -------------------------------------------------- | ------------------------------------------------------------------ |
| `src/credit_default_risk/data.py`                | Raw table cleaning, history aggregation, train/test split          |
| `src/credit_default_risk/features.py`            | Shared feature engineering and CatBoost input preparation          |
| `src/credit_default_risk/train.py`               | Final model fit, evaluation, serialization, MLflow registration    |
| `src/credit_default_risk/predict.py`             | Single and batch inference from engineered records                 |
| `src/credit_default_risk/helper.py`              | Reusable transformations and feature selection                     |
| `src/credit_default_risk/config.py`              | Settings and feature definitions                                   |
| `src/credit_default_risk/logging_conf.py`        | Structured JSON logging and correlation ID context                 |
| `src/credit_default_risk/api.py`                 | Placeholder for the future FastAPI app                             |
| `data/raw`, `data/interim`, `data/processed` | Input and generated data; contents are Git ignored                 |
| `artifacts/catboost_final`                       | Local model bundle; contents are Git ignored                       |
| `notebooks/`                                     | Research and model selection notebooks                             |
| `reports/final_model/`                           | Tracked report and inference contract for an earlier model version |
| `pyproject.toml`, `uv.lock`                    | Package metadata, dependencies, and resolved environment           |

## Development checks

```bash
uv run --extra dev pre-commit run --all-files
```

The installed pre-commit hook also runs Ruff, Black, and the end-of-file fixer on staged files at commit time. The first hook run may download its pinned tool environments.

Full Running Commands:

´´´

pip install-e".[dev]"#install
ruffchecksrctests&&black--checksrctests#lint
pytest-v--cov=src/prodml--cov-report=term-missing#test
python-mprodml.train#train
uvicornprodml.api.main:app--reload--port8000#serve

´´´

```Shell
uv run uvicorn credit_default_risk.api.main:app --reload --port 8000
```



```Shell
uv run --extra dev pytest -q
```
