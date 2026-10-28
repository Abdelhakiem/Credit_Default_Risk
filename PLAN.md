# Credit Default Risk Project — TODO

## Phase 1 — Initial data exploration, cleaning, preprocessing, and EDA

### 1. Define the prediction problem

- [ ] Record the prediction unit (for example, applicant, account, or account-month) and the population eligible for scoring.
- [ ] Define the prediction timestamp, default event, performance window, and outcome-maturity period with the business owner.
- [ ] Specify the positive class and treatment of unresolved, censored, or not-yet-matured outcomes. Do not label immature outcomes as non-defaults.
- [ ] Set a feature availability cutoff: every input must be known at scoring time. Flag post-decision, post-delinquency, recovery, and collection fields as potential leakage.
- [ ] Create a data dictionary with business definitions, units, source systems, and expected ranges for each field.
- [ ] Record data-use constraints, including whether sensitive or protected attributes may be used for modeling and/or fairness evaluation.

### 2. Inspect and profile the raw data

- [ ] Keep source files immutable in `data/raw/`; record source, retrieval date, file checksums, and schema/version.
- [ ] Capture row and column counts, column names, inferred types, memory use, date ranges, and target prevalence.
- [ ] Identify keys, entity IDs, timestamps, repeated observations, duplicate rows, and duplicate entity-period records.
- [ ] Profile null counts and rates, sentinel values (such as `-1`, `999`, or `Unknown`), category cardinality, and numeric ranges.
- [ ] Check target values, missing labels, class balance, and whether every label has had enough time to mature.
- [ ] Save a profiling report and document anomalies instead of silently correcting them.

### 3. Establish data quality rules and clean safely

- [ ] Validate required columns, types, uniqueness expectations, allowed category values, and business-rule ranges.
- [ ] Resolve duplicate records using a documented rule; never drop rows solely because a row looks unusual.
- [ ] Standardize types, dates, whitespace, category spelling, and explicit missing-value codes.
- [ ] Separate invalid/impossible values from plausible extreme values; investigate extremes before deciding to correct, cap, or retain them.
- [ ] Assess whether missingness is meaningful (for example, “no prior loan”) versus unknown or a data-pipeline failure.
- [ ] Keep an audit trail of excluded rows and cleaning decisions, including counts and reasons.
- [ ] Add checks that fail clearly when the incoming schema or basic data assumptions change.

### 4. Design and freeze the data split

- [ ] Choose the split based on how the model will be used. Prefer an out-of-time test set when deployment predicts future applications or periods.
- [ ] Keep all records for the same borrower/account on one side of a split when repeated entities could otherwise leak across sets.
- [ ] Use a stratified random split only when observations are plausibly independent and there is no meaningful time ordering.
- [ ] Create train, validation, and test partitions before fitting imputers, encoders, feature selection, or other learned transforms.
- [ ] Check split sizes, target rates, time coverage, entity overlap, and key population differences; document any trade-offs.
- [ ] Freeze the test set for final evaluation. Use train for fitting and validation for model/threshold choices.
- [ ] Save a reproducible split manifest (row/entity keys, split seed or date boundaries, and data version), without copying sensitive data unnecessarily.

### 5. Explore the data (EDA)

- [ ] Plot target prevalence overall and by time/cohort, product, channel, geography, and other permitted operational segments.
- [ ] Inspect numeric distributions, skew, ranges, zero-inflation, outliers, and missingness; use scales or bins that make long tails readable.
- [ ] Inspect categorical frequencies, rare levels, inconsistent labels, and missingness.
- [ ] Compare feature distributions and missingness across train, validation, test, and time cohorts.
- [ ] Examine default rates by sensible numeric bins and categorical levels, with sample counts and uncertainty so small groups are not over-read.
- [ ] Check pairwise correlations and redundant features; use these as investigation prompts, not automatic deletion rules.
- [ ] Review candidate features for leakage, unstable definitions, post-outcome timing, and dependence on operational processes.
- [ ] Examine performance and data coverage across relevant customer groups; document fairness questions for review with the appropriate stakeholders.
- [ ] Summarize findings, data limitations, and unresolved questions in a concise EDA report.

### 6. Specify baseline preprocessing

- [ ] Build preprocessing as a scikit-learn `Pipeline`/`ColumnTransformer` so transforms are fit on training data only and reused consistently at inference.
- [ ] Numeric baseline: type validation, train-fitted imputation, and optional missingness indicators where useful.
- [ ] Categorical baseline: explicit missing handling, train-fitted one-hot encoding, and a defined policy for unseen categories.
- [ ] Decide whether scaling or transformations are needed for the selected model family; tree models may not need scaling.
- [ ] Handle rare/high-cardinality categories only after measuring their frequency and considering inference-time behavior.
- [ ] Keep feature selection and any resampling/class-weight decisions inside the training workflow; never resample validation or test data.
- [ ] Confirm output feature names, order, types, and behavior on rows with missing values or unseen categories.

### 7. Phase 1 deliverables / exit criteria

- [ ] Data dictionary and target/label definition approved for the chosen use case.
- [ ] Data quality and leakage findings documented, with explicit decisions for each material issue.
- [ ] Reproducible train/validation/test split created and split diagnostics reviewed.
- [ ] EDA report covers target balance, feature distributions, missingness, temporal stability, and key risks.
- [ ] Baseline preprocessing pipeline can fit on train and transform validation/test without refitting or schema surprises.
- [ ] Open decisions and assumptions are captured before feature engineering and model comparison begin.

## Later phases — after Phase 1

### Feature engineering and training workflow

- [ ] Engineer only features available at the prediction timestamp; document definitions and expected freshness.
- [ ] Turn approved feature transformations into reusable, tested pipeline components.
- [ ] Add DVC stages for data validation, feature creation, training, and evaluation once the workflow is stable.

### Modeling and evaluation

- [ ] Establish a simple baseline, then compare suitable models using the frozen split strategy.
- [ ] Choose metrics tied to the use case: discrimination (ROC AUC/PR AUC), ranking/lift, calibration, and decision-cost or approval-rate analysis as applicable.
- [ ] Report uncertainty and segment/time stability; choose operating thresholds with business costs and constraints in view.
- [ ] Add explainability appropriate to the model and audience; review feature effects for leakage and implausible behavior.
- [ ] Evaluate and, if justified, calibrate probabilities using validation data; reserve test data for the final report.
- [ ] Track data, code, parameters, metrics, and artifacts with MLflow.

### Inference and operations

- [ ] Define a versioned inference input/output schema and prediction-time validation.
- [ ] Implement the FastAPI service and package it with Docker.
- [ ] Add structured logs, latency/error monitoring, data-quality checks, drift and calibration monitoring, and model/version tracking.
- [ ] Document deployment, rollback, retraining triggers, and model limitations.
