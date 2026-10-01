# Predicting school attendance in nine districts of Pakistan (PSLM 2019-20)

Code, data and results for the manuscript *Predictive Modelling for Educational Access: A
Scalable, Explainable AI Framework for Evidence-Based Policy in Low-Resource Settings*.

The repository rebuilds every number in the manuscript from the raw survey files: the
analysis dataset, the descriptive statistics, the model comparison (Table 2), the predictor
rankings (Tables 3 and B1), the disability analysis (Table 4) and the appendix tables (A1, A3,
D1).

---

## 1. Research question

**Can general-purpose household surveys, analysed with interpretable machine learning,
identify which children are out of school and what distinguishes them, separately for
districts that differ widely in deprivation?**

The study uses data that are already collected (the Pakistan Social and Living Standards
Measurement Survey, PSLM) rather than a purpose-built instrument, and fits the same analysis in
nine districts: the federal capital, the four provincial capitals, and one high-deprivation
district per province.

### Hypotheses

| | Hypothesis | How it is tested |
|---|---|---|
| **H1** | Flexible models (random forest, neural network) distinguish attending from out-of-school children better than a main-effects logistic regression. | AUC-ROC of the best machine-learning model against logistic regression in each district; 95 % CIs and three paired tests with Holm correction across the nine districts (Table 2); robustness in four sensitivity analyses (Table A3). |
| **H2** | The leading predictors of attendance differ between the large cities and the high-deprivation districts. | Out-of-fold SHAP rankings of the random forest per district (Tables 3 and B1), with their agreement with logistic-regression rankings. |

A descriptive question is examined alongside: how attendance differs for children with a
functional difficulty, and whether the gap varies with the education of the household head
(Table 4).

### Results in brief

- **H1 supported.** The random forest is the best model in all nine districts and outperforms
  logistic regression in every district (mean AUC 0.828 against 0.768; gains 0.026 to 0.117).
  All differences remain significant after Holm correction under all three tests, and the
  gain holds in every sensitivity analysis. The neural network performs close to logistic
  regression.
- **H2 partly supported.** The education of the household head is among the five strongest
  predictors in every district. Beyond it, household internet access ranks second or third in
  the three largest cities; toilet type ranks among the eight strongest predictors in all four
  high-deprivation districts; gender is a leading predictor in Quetta, Peshawar and the four
  high-deprivation districts but not in the large cities.
- **Disability.** Children with a functional difficulty attend less often (54.6 % against
  67.7 %) at every level of household-head education.

---

## 2. Data

| | |
|---|---|
| Survey | Pakistan Social and Living Standards Measurement Survey (PSLM) 2019-20, Pakistan Bureau of Statistics |
| Files used | Stata modules `roster`, `plist`, `secb2`, `secc1`, `secc2`, `secd`, `sece`, `secf1`, `secf2`, `secg`, `seck`, `secl` |
| Districts | Karachi Central, Islamabad, Lahore, Quetta, Peshawar (capitals); Kohistan, Dera Bugti, Rajanpur, Tharparkar (high deprivation) |
| Unit of analysis | Child aged 5-16 |
| Sample | 30,592 children in 11,992 households |
| Outcome | Currently attending school (1) or not (0) |

The raw PSLM files are not included. They are available from the Pakistan Bureau of
Statistics and go in `raw_data/stata data/`. The analysis dataset built from them
(`data/analysis_data.parquet`) **is** included, so notebooks 02 and 03 run without the raw
files.

---

## 3. Analysis pipeline

```
raw PSLM Stata files
      │  notebook 01  Data preparation      → data/analysis_data.parquet, data/variables.csv
      ▼
analysis dataset (one row per child)
      │  notebook 02  Descriptive analysis  → results/descriptives/*.csv, Tables D1 and 4
      ▼
      │  notebook 03  Models                → Tables 2, A1, A3, B1 (and Table 3)
      ▼
results/tables/*.csv
```

Each notebook is organised in numbered steps, one step per block, each followed by its
output.

### Step 1 · Data preparation (`notebooks/01_data_preparation.ipynb`, `src/build_dataset.py`)

**1.1 Study districts.** Nine districts selected on the Multidimensional Poverty Index (MPI):
the five capitals and the most deprived district of each province.

**1.2 Sample.** All usual household members (roster `sb1q11 = 1`) aged 5 to 16 in the nine
districts, with the household sampling weight and primary sampling unit.

**1.3 Outcome.** `education_access = 1` if the child is currently attending school
(`secc1 sc1q01 = 3`); `0` if the child never attended (`= 1`) or attended in the past (`= 2`).

**1.4 Linking modules.** Person-level modules (`secb2`, `secc1`, `secc2`, `secd`, `sece`)
are matched to the child by household and person code; household-level modules (`secf1`,
`secf2`, `secg`, `seck`, `secl`) by household code, so siblings share their household's
values.

**1.5 Predictors.** 56 candidate predictors (`src/config.py`), each labelled as measured for
the child (27) or for the household (29).

- *48 survey questions used as recorded.* Coded answers become text categories using the
  PSLM value labels; survey codes without a numeric meaning (for example `98` don't know and
  `99` refused in the food-insecurity questions) are kept as categories, never as numbers.
  Eight predictors are numeric (age, number of rooms, head age, head education level, total
  assets, log household income, work days, monthly earnings).
- *8 constructed variables:*

  | Variable | Construction |
  |---|---|
  | Disability | "Some difficulty" or worse in any of the six Washington Group domains (`secb2 sb2q06`–`sb2q11`) |
  | Head age | Age of the household head (first listed head; 5 households list two). Age 99 ("not known") is set to missing |
  | Head gender | Sex of the household head |
  | Head education level | 0 never attended; 1 classes 1-4; 2 class 5; 3 classes 6-12; 4 degree; 5 other. Heads still studying are missing |
  | Total assets | Number of property and livestock items owned (`secg` items 1, 4-10) |
  | Property owner gender | Sex of the owner of agricultural land (`secg` item 1) |
  | Transport mode | Mode of transport to the nearest basic health unit (`secl`) |
  | Household income (log) | Annual income of all members from all sources: main job (monthly pay × months worked, or annual pay), other work, in-kind wages sold, pensions, domestic transfers, remittances, rent and other income. Zero only when nobody in the household worked; missing when someone worked but no cash amount was recorded. Entered as log(1 + income) |

- *School-dependent items.* Three questions are recorded only for children who attend or
  attended school (can do math, can write, years to complete primary). They partly encode the
  outcome and are excluded from the models (used only in a sensitivity analysis).

**1.6 Quality checks.** One row per child; ages within 5-16; outcome coded 0/1; every child
has a sampling weight; distribution of household-income status; share of missing values per
predictor. No values are imputed at this stage: missing values are kept and handled inside
the cross-validation (Step 3.2).

### Step 2 · Descriptive analysis (`notebooks/02_descriptive_analysis.ipynb`)

1. Sample size, households, PSUs and share of children out of school per district
   (unweighted and survey-weighted).
2. Attendance by sex, age group and region.
3. Missing values per predictor and district.
4. Association of each predictor with attendance (|r| for numeric, Cramér's V for categorical).
5. Evidence that the three school-dependent items are recorded almost only for children in
   school.
6. **Table D1:** predictors used in all districts, in some districts, or in none.
7. **Table 4:** attendance by functional difficulty and household-head education (nine
   districts pooled; 95 % CIs from a bootstrap of households), plus attendance by age, sex
   and disability per district.

### Step 3 · Models (`notebooks/03_models.ipynb`, `src/modeling.py`, `src/evaluation.py`)

**3.1 Models** (settings fixed in advance, identical in every district, not tuned):

| Model | Settings |
|---|---|
| Logistic regression (baseline) | L2 penalty, C = 1, liblinear, balanced class weights, standardised inputs |
| Random forest | 300 trees, maximum depth 15, minimum 5 samples to split and 2 per leaf, √p predictors per split, balanced class weights |
| Neural network (Keras) | Standardised inputs; 128 ReLU units + batch normalisation + dropout 0.3; 64 ReLU units + dropout 0.3; sigmoid output; Adam (learning rate 0.001); binary cross-entropy with class weights; 30 epochs, batch size 32 |

**3.2 Preprocessing**, learned from the training folds only and then applied to the test fold:

1. drop predictors missing for more than 50 % of training children;
2. impute the median (numeric) or the most frequent category (categorical);
3. one-hot encode categorical predictors;
4. scale all columns to [0, 1];
5. keep columns with |correlation| ≥ 0.01 with the outcome in the training data.

**3.3 Validation.** Models are fitted separately in each district. Stratified five-fold
cross-validation, repeated five times with different partitions (25 test sets of about 20 %
of the children). All children of a household are assigned to the same fold
(`StratifiedGroupKFold`), so siblings never appear in both training and test data.

**3.4 Metrics.** AUC-ROC (primary) and survey-weighted AUC; precision, recall, F1 and
PR-AUC for the out-of-school class; accuracy and F1 for attending children. Threshold 0.5 for
threshold-based metrics.

**3.5 Model comparison (H1).** For each district, the best machine-learning model against
logistic regression:

- 95 % CIs and the paired difference over the 25 test sets with the corrected resampled
  t-test (Nadeau & Bengio, 2003);
- DeLong test on the out-of-fold predictions (DeLong et al., 1988; Sun & Xu, 2014);
- household-cluster bootstrap of the AUC difference (2,000 resamples of households);
- Holm correction across the nine districts; a difference is called significant only when
  all three tests agree.

**3.6 Sensitivity analyses** (three repetitions each):

| Analysis | Fold assignment | School-dependent items | Child's own work and digital use |
|---|---|---|---|
| Primary | households | excluded | included |
| `child_split` | individual children | excluded | included |
| `with_school_vars` | households | included | included |
| `with_school_vars_child_split` | individual children | included | included |
| `without_concurrent` | households | excluded | excluded |

**3.7 Interpretation (H2).** SHAP values of the random forest (TreeExplainer), computed
out-of-fold in the first repetition: every child is explained by a model that did not see
that child's household. Values of one-hot columns are summed back to the original predictor.
Logistic regression is explained in the same way (LinearExplainer), and the two rankings are
compared with Spearman's rank correlation.

---

## 4. Assumptions

| Assumption | Why it is made | How it is checked or qualified |
|---|---|---|
| The outcome is a household report of current attendance at the interview. | It is the attendance measure recorded for every child. | It does not measure regularity of attendance or learning; stated as a limitation. |
| Predictors describe the child's circumstances, not consequences of attendance. | Required for a predictive model to be meaningful. | School-dependent items excluded; the child's own work and digital use tested in a sensitivity analysis. |
| Age is a legitimate predictor. | Fixed before the attendance decision; marks late entry and dropout. | Ranked but not interpreted as a cause. |
| Children are dependent within households, independent across households. | Siblings share household predictors. | Household-grouped folds and household-cluster bootstrap. Clustering by PSU is not modelled. |
| Missing values are handled within the training data. | Avoids information from test children. | Predictors missing for over half the training children are dropped; the rest imputed. |
| Models are trained without sampling weights. | The aim is prediction within each district sample, not population totals. | Survey-weighted AUC reported alongside. |
| One set of model settings for all districts. | Avoids tuning on test data and keeps districts comparable. | Settings listed in Section 3.1. |
| SHAP values describe associations learned by the model. | SHAP explains predictions, not causes. | Rankings compared across two model families; no causal claims. |

---

## 5. Outputs and the manuscript

| Manuscript | File | Produced by |
|---|---|---|
| Sample and prevalence (Section 3.3) | `results/descriptives/sample_and_prevalence.csv` | 02 |
| Age, sex and disability figures in the text | `results/descriptives/attendance_by_age.csv`, `attendance_by_sex.csv`, `attendance_by_disability.csv` | 02 |
| Table 2 | `results/tables/table2_performance.csv` | 03 |
| Table 3 (ranks 1-5 of Table B1) | `results/tables/tableB1_top_predictors.csv` | 03 |
| Table 4 | `results/tables/table4_disability_by_head_education.csv` | 02 |
| Table A1 | `results/tables/tableA1_all_metrics.csv` | 03 |
| Table A3 | `results/tables/sensitivity_summary.csv`, `all_test_sets.csv` | 03 |
| Table B1 | `results/tables/tableB1_top_predictors.csv` | 03 |
| Predictor ranks cited in the text | `results/tables/predictor_ranks.csv` | 03 |
| Table D1 | `results/tables/tableD1_predictors.csv`, `tableD1_predictors_long.csv` | 02 |
| Variable list | `data/variables.csv` | 01 |

---

## 6. Repository structure

```
notebooks/
  01_data_preparation.ipynb      raw PSLM files → analysis dataset
  02_descriptive_analysis.ipynb  sample, predictors, Tables D1 and 4
  03_models.ipynb                cross-validation, tests, SHAP; Tables 2, A1, A3, B1
src/
  config.py         districts, predictors (source, definition, level), modelling rules
  build_dataset.py  construction of the analysis dataset
  modeling.py       preprocessing, models, cross-validation, SHAP
  evaluation.py     metrics, corrected resampled t-test, DeLong test, cluster bootstrap, Holm
data/               analysis dataset and variable list
results/            descriptives/ and tables/ (results/cv/ holds cached predictions, not tracked)
raw_data/           PSLM 2019-20 Stata files (not included)
requirements.txt    exact package versions
```

---

## 7. Reproduce

Requires Python 3.12.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt          # on macOS/Linux: .venv/bin/pip
.venv/Scripts/python -m ipykernel install --user --name district_analysis
# optional: place the PSLM 2019-20 .dta files in raw_data/stata data/ to rebuild the dataset
cd notebooks
../.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace 01_data_preparation.ipynb
../.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace 02_descriptive_analysis.ipynb
../.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace 03_models.ipynb
```

Skip notebook 01 to start from the included dataset.

**Runtime.** The three notebooks take under one hour on 8 CPU cores (no GPU). Notebook 03
caches its cross-validation predictions in `results/cv/`, after which it re-runs in minutes.

**Reproducibility.** Random seed 42 throughout. Logistic regression and random forest results
are exactly reproducible; neural-network results can differ between runs in the fourth decimal
place because of floating-point non-determinism in parallel execution.

**Windows.** TensorFlow must be loaded before pandas; the notebooks do this by importing `src`
first.

---

## 8. Limitations

- Observational data: SHAP rankings describe predictive associations, not causal effects.
- Nine districts, purposively selected; results describe the surveyed samples (unweighted
  training).
- Few children with a functional difficulty per district, so the disability analysis pools
  the districts.
- The outcome is current attendance as reported by the household.
- The MICS validation reported in the manuscript (Lahore District) is a separate analysis and
  is not part of this repository.
