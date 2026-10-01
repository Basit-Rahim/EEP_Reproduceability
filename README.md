# Predicting school attendance in nine districts of Pakistan (PSLM 2019-20)

Code and results for *Predictive Modelling for Educational Access: A Scalable, Explainable
AI Framework for Evidence-Based Policy in Low-Resource Settings*.

Children aged 5-16 in nine districts of the Pakistan Social and Living Standards
Measurement Survey (PSLM) 2019-20. Outcome: currently attending school. Models: Logistic
Regression (baseline), Random Forest and a feed-forward Neural Network, fitted separately in
each district and interpreted with SHAP.

## Structure

```
notebooks/
  01_data_preparation.ipynb      PSLM Stata files -> child-level analysis dataset
  02_descriptive_analysis.ipynb  prevalence, missing values, associations, Table D1
  03_models.ipynb                cross-validation, Tables 2, A1, B1, sensitivity analyses
src/
  config.py         districts, predictors (source and definition), modelling rules
  build_dataset.py  construction of the analysis dataset
  modeling.py       preprocessing, models, cross-validation, SHAP
  evaluation.py     metrics, corrected resampled t-test, DeLong test, household-cluster
                    bootstrap, Holm correction
data/             analysis dataset and variable list (notebook 01)
results/          tables (notebooks 02 and 03); results/cv holds cached predictions
raw_data/         PSLM 2019-20 Stata files (from the Pakistan Bureau of Statistics; not included)
```

## Reproduce

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m ipykernel install --user --name district_analysis
# place the PSLM 2019-20 .dta files in raw_data/stata data/
cd notebooks
../.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace 01_data_preparation.ipynb
../.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace 02_descriptive_analysis.ipynb
../.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace 03_models.ipynb
```

The analysis dataset built by notebook 01 (`data/analysis_data.parquet`, derived from the
PSLM 2019-20 microdata of the Pakistan Bureau of Statistics) is included, so notebooks 02
and 03 can be run without the raw survey files.

On Windows, TensorFlow must be loaded before pandas; the notebooks do this by importing
`src` first. The three notebooks take under one hour on 8 cores; notebook 03 then re-runs in minutes
from the cached predictions. Logistic regression and random forest results are exactly
reproducible; neural-network results can differ between runs in the fourth decimal place
(floating-point non-determinism in parallel execution).

## Methods in brief

- **Sample:** usual household members aged 5-16 (30,592 children in 11,992 households).
- **Predictors:** 56 candidates measured for the child or the household (survey questions
  and constructed variables, see `src/config.py`); household income enters as
  log(1 + income). Predictors recorded only for children who attend school (can do math,
  can write, years to complete primary) are excluded. Within each training set, predictors
  missing for more than 50 % of children are dropped.
- **Preprocessing** (learned from training data only): median / most-frequent imputation,
  one-hot encoding, scaling to [0, 1], indicator columns with |r| >= 0.01 with the outcome.
- **Models:** Logistic Regression (L2, C = 1, class-balanced); Random Forest (300 trees,
  depth 15, class-balanced); Neural Network (128-64, batch normalisation, dropout 0.3,
  30 epochs, class weights).
- **Validation:** stratified 5-fold cross-validation repeated 5 times; all children of a
  household in the same fold.
- **Inference:** 95 % confidence intervals and paired model comparisons by the corrected
  resampled t-test; DeLong test and household-cluster bootstrap on out-of-fold predictions;
  Holm correction across the nine districts.
- **Sensitivity analyses:** children assigned to folds at random; school-dependent
  predictors included; the child's own work and computer / mobile use (measured at the same
  time as attendance) excluded.
- **Interpretation:** SHAP values of the Random Forest (TreeExplainer), out-of-fold, with
  agreement to logistic-regression SHAP rankings.

Training is unweighted (prediction task); the survey-weighted AUC is reported alongside.
