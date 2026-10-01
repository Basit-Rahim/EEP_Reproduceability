"""
Models and cross-validation.

Preprocessing (learned from the training data of each fold only)
  1. drop predictors missing for more than 50 % of training children
  2. impute: median (numeric), most frequent category (categorical)
  3. one-hot encode categorical predictors
  4. scale to [0, 1]
  5. keep indicator columns with |correlation| >= 0.01 with the outcome

Models (identical settings in every district)
  * Logistic Regression: L2 penalty, C = 1, liblinear, class-balanced, standardised inputs
  * Random Forest: 300 trees, max depth 15, min split 5, min leaf 2, sqrt features,
    class-balanced
  * Neural Network (Keras): 128 ReLU + batch normalisation + dropout 0.3, 64 ReLU +
    dropout 0.3, sigmoid output; Adam (0.001), binary cross-entropy, 30 epochs,
    batch 32, class weights; seeded

Validation: 5-fold cross-validation repeated with different shuffles; folds keep
all children of a household together.
"""

import multiprocessing
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from .config import MAX_MISSING_SHARE, MIN_ABS_CORRELATION, N_FOLDS, SEED

MODELS = ["LR", "RF", "NN"]
MODEL_NAMES = {"LR": "Logistic Regression", "RF": "Random Forest", "NN": "Neural Network"}


class Preprocessor(BaseEstimator, TransformerMixin):
    """Steps 1-5 above, learned from the training data only."""

    def __init__(self, max_missing_share=MAX_MISSING_SHARE, min_abs_corr=MIN_ABS_CORRELATION):
        self.max_missing_share = max_missing_share
        self.min_abs_corr = min_abs_corr

    def fit(self, X, y):
        X = X.copy()
        # 1. drop predictors missing for too many training children
        self.kept_ = [c for c in X.columns if X[c].isna().mean() <= self.max_missing_share]
        X = X[self.kept_]
        self.numeric_ = [c for c in self.kept_ if pd.api.types.is_numeric_dtype(X[c])]
        self.categorical_ = [c for c in self.kept_ if c not in self.numeric_]
        # 2. impute: median (numeric), most frequent value (categorical)
        self.fill_ = {c: X[c].median() for c in self.numeric_}
        self.fill_.update({c: X[c].mode().iloc[0] if X[c].notna().any() else "missing"
                           for c in self.categorical_})
        # 3. one-hot encode categorical predictors (levels seen in training)
        self.levels_ = {c: sorted(X[c].fillna(self.fill_[c]).astype(str).unique())
                        for c in self.categorical_}
        Z = self._encode(X)
        # 4. MinMax scaling
        self.min_, self.range_ = Z.min(), (Z.max() - Z.min()).replace(0, 1)
        Z = (Z - self.min_) / self.range_
        # 5. correlation filter with the outcome
        corr = Z.corrwith(pd.Series(np.asarray(y), index=Z.index))
        self.columns_ = corr[corr.abs() >= self.min_abs_corr].index.tolist()
        self.source_ = {col: self._source_of(col) for col in self.columns_}
        return self

    def _encode(self, X):
        parts = [X[self.numeric_].fillna(self.fill_).astype(float)] if self.numeric_ else []
        for c in self.categorical_:
            v = X[c].fillna(self.fill_[c]).astype(str)
            parts.append(pd.DataFrame({f"{c}_{lvl}": (v == lvl).astype(float)
                                       for lvl in self.levels_[c]}, index=X.index))
        return pd.concat(parts, axis=1)

    def _source_of(self, col):
        if col in self.numeric_:
            return col
        return max((c for c in self.categorical_ if col.startswith(c + "_")), key=len)

    def transform(self, X):
        Z = self._encode(X[self.kept_])
        Z = (Z - self.min_) / self.range_
        return Z.reindex(columns=self.columns_, fill_value=0.0)


class KerasNN(BaseEstimator):
    """Feed-forward network (see module docstring), seeded for reproducibility."""

    def __init__(self, epochs=30, batch_size=32, seed=SEED):
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, X, y):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
        from tensorflow.keras.optimizers import Adam

        try:  # few threads per process: several folds run in parallel
            tf.config.threading.set_intra_op_parallelism_threads(2)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except RuntimeError:
            pass  # already initialised in this process
        tf.keras.utils.set_random_seed(self.seed)
        tf.config.experimental.enable_op_determinism()
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        y = np.asarray(y)
        classes = np.unique(y)
        weights = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=y)))
        self.model_ = Sequential([
            Input(shape=(Xs.shape[1],)),
            Dense(128, activation="relu"), BatchNormalization(), Dropout(0.3),
            Dense(64, activation="relu"), Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ])
        self.model_.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy")
        self.model_.fit(Xs, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0,
                        class_weight=weights)
        return self

    def predict_proba(self, X):
        p = self.model_.predict(self.scaler_.transform(X), verbose=0).ravel()
        return np.column_stack([1 - p, p])


def make_model(name, seed=SEED):
    pre = Preprocessor()
    if name == "LR":
        clf = Pipeline([("scale", StandardScaler()),
                        ("lr", LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                                  max_iter=1000, class_weight="balanced"))])
    elif name == "RF":
        clf = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=5,
                                     min_samples_leaf=2, max_features="sqrt",
                                     class_weight="balanced", random_state=seed, n_jobs=1)
    else:
        clf = KerasNN(seed=seed)
    return Pipeline([("pre", pre), ("model", clf)])


# ---------------------------------------------------------------------------
def splits(y, groups, household, repeat):
    """Stratified 5-fold split (80/20): households kept together (household=True), or
    children assigned at random (household=False, sensitivity analysis)."""
    if household:
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + repeat)
        return list(cv.split(np.zeros(len(y)), y, groups))
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + repeat)
    return list(cv.split(np.zeros(len(y)), y))


def _fit_fold(X, y, train, test, seed, explain):
    warnings.filterwarnings("ignore")
    out = {"test": test, "proba": {}, "shap": None}
    fitted = {}
    for name in MODELS:
        model = make_model(name, seed).fit(X.iloc[train], y[train])
        out["proba"][name] = model.predict_proba(X.iloc[test])[:, 1]
        fitted[name] = model
    if explain:
        out["shap"] = explain_fold(fitted, X.iloc[train], X.iloc[test])
    return out


def explain_fold(fitted, X_train, X_test):
    """SHAP values summed back to the predictors: RF (TreeExplainer) and
    LR (LinearExplainer), for the children in this test fold."""
    import shap

    result = {}
    rf = fitted["RF"]
    pre, forest = rf.named_steps["pre"], rf.named_steps["model"]
    Z = pre.transform(X_test)
    sv = shap.TreeExplainer(forest).shap_values(Z, check_additivity=False)
    sv = sv[:, :, 1] if np.ndim(sv) == 3 else sv[1] if isinstance(sv, list) else sv
    result["RF"] = pd.DataFrame(sv, columns=Z.columns, index=X_test.index) \
        .T.groupby(pre.source_).sum().T

    lr = fitted["LR"]
    pre, lr_pipe = lr.named_steps["pre"], lr.named_steps["model"]
    Ztr, Zte = pre.transform(X_train), pre.transform(X_test)
    scaler, logit = lr_pipe.named_steps["scale"], lr_pipe.named_steps["lr"]
    lsv = shap.LinearExplainer(logit, scaler.transform(Ztr)).shap_values(scaler.transform(Zte))
    result["LR"] = pd.DataFrame(lsv, columns=Zte.columns, index=X_test.index) \
        .T.groupby(pre.source_).sum().T
    return result


def run_cv(df, features, target, household, n_repeats, explain_first_repeat=False,
           n_jobs=8):
    """Repeated 5-fold cross-validation. Returns one dict per (repeat, fold) with the
    test indices and each model's predicted probability of attending."""
    X = df[features].reset_index(drop=True)
    y = df[target].to_numpy()
    groups = df["hhcode"].to_numpy()
    tasks = [(r, f, tr, te) for r in range(n_repeats)
             for f, (tr, te) in enumerate(splits(y, groups, household, r))]
    # A fresh process per fold: TensorFlow can deadlock in reused worker processes.
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx,
                             max_tasks_per_child=1) as pool:
        futures = [pool.submit(_fit_fold, X, y, tr, te, SEED + r,
                               explain_first_repeat and r == 0)
                   for r, f, tr, te in tasks]
        results = [fut.result() for fut in futures]
    for (r, f, _, _), res in zip(tasks, results):
        res.update(repeat=r, fold=f)
    return results
