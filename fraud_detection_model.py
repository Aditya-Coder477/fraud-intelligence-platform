"""
================================================================================
  MULE ACCOUNT DETECTION — STATE-OF-THE-ART FRAUD DETECTION MODEL
  Target    : F3924  (0 = Legitimate, 1 = Mule Account)
  Imbalance : 99.11% / 0.89%  (112:1 negative-to-positive ratio)
  Models    : LightGBM · XGBoost · CatBoost · Stacking Ensemble
================================================================================

PIPELINE OVERVIEW
─────────────────
  1.  Data loading & preprocessing (NaN fill, type enforcement)
  2.  Imbalance handling  → class weights + SMOTE on training folds only
  3.  Stratified 5-Fold cross-validation
  4.  Per-model hyperparameter optimisation  (Optuna, 50 trials each)
  5.  OOF (Out-Of-Fold) predictions → unbiased metrics
  6.  Threshold calibration  (maximise F1, target Recall ≥ 0.85)
  7.  Stacking meta-learner  (Logistic Regression on OOF probabilities)
  8.  SHAP explanations       (global importance + per-instance waterfall)
  9.  Model persistence       (best model + ensemble + feature list)
  10. Inference pipeline       (MuleAccountInferencePipeline class)

OUTPUTS
───────
  models/
    lgbm_best.pkl          LightGBM final model (full-data retrain)
    xgb_best.pkl           XGBoost final model
    catboost_best.pkl      CatBoost final model
    ensemble_meta.pkl      Stacking meta-learner
    best_model.pkl         Single best model by PR-AUC
    inference_pipeline.pkl End-to-end inference pipeline
    thresholds.json        Optimal classification thresholds
    cv_results.json        Full cross-validation metrics
  plots/
    shap_beeswarm.png      Global SHAP beeswarm (top-30)
    shap_bar.png           Mean |SHAP| bar chart
    pr_curve.png           Precision-Recall curves (all models)
    roc_curve.png          ROC curves (all models)
    confusion_matrices.png Confusion matrices at optimal threshold
    calibration_curve.png  Probability calibration plot
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import os
import sys
import json
import warnings
import logging
import re
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("MuleDetector")

# ── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, recall_score, precision_score, roc_auc_score,
    average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve,
    brier_score_loss,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    log.warning("imbalanced-learn not installed — SMOTE disabled. "
                "Install: pip install imbalanced-learn")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    log.warning("Optuna not installed — using default hyperparameters. "
                "Install: pip install optuna")

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import shap

# ════════════════════════════════════════════════════════════════════════════
#  CONSTANTS & CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
TARGET          = "F3924"
RANDOM_STATE    = 42
N_FOLDS         = 5
OPTUNA_TRIALS   = 50          # increase to 100+ for production
MIN_RECALL      = 0.80        # minimum recall at threshold selection
SMOTE_RATIO     = 0.15        # SMOTE minority-class target ratio (of majority)

# Directories
BASE_DIR   = Path(".")
MODEL_DIR  = BASE_DIR / "models"
PLOT_DIR   = BASE_DIR / "plots"
MODEL_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 0 — UTILITIES
# ════════════════════════════════════════════════════════════════════════════
def sanitize_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """Replace all non-alphanumeric/underscore characters in column names."""
    df.columns = [re.sub(r"[^A-Za-z0-9_]", "_", c) for c in df.columns]
    # Deduplicate after sanitisation
    seen = {}
    new_cols = []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    df.columns = new_cols
    return df


def compute_class_weight(y: np.ndarray) -> float:
    """Return scale_pos_weight = n_neg / n_pos for binary classifiers."""
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return float(n_neg / max(n_pos, 1))


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_recall: float = MIN_RECALL,
) -> dict:
    """
    Find the probability threshold that maximises F1-score while keeping
    Recall ≥ min_recall.  Falls back to the best-F1 threshold if the
    recall constraint cannot be met.
    """
    precs, recs, threshs = precision_recall_curve(y_true, y_prob)
    # threshs has len = len(precs) - 1
    f1s = 2 * precs[:-1] * recs[:-1] / (precs[:-1] + recs[:-1] + 1e-9)

    # Filter to thresholds meeting recall constraint
    valid = recs[:-1] >= min_recall
    if valid.any():
        idx = np.argmax(f1s * valid)
    else:
        idx = np.argmax(f1s)

    return {
        "threshold":    float(threshs[idx]),
        "f1":           float(f1s[idx]),
        "precision":    float(precs[idx]),
        "recall":       float(recs[idx]),
        "pr_auc":       float(average_precision_score(y_true, y_prob)),
        "roc_auc":      float(roc_auc_score(y_true, y_prob)),
    }


def apply_smote(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    ratio: float = SMOTE_RATIO,
) -> tuple:
    """Apply SMOTE only if available and positive class is small."""
    if not SMOTE_AVAILABLE or y_tr.sum() < 5:
        return X_tr, y_tr
    try:
        sm = SMOTE(
            sampling_strategy=ratio,
            random_state=RANDOM_STATE,
            k_neighbors=min(5, int(y_tr.sum()) - 1),
        )
        X_res, y_res = sm.fit_resample(X_tr, y_tr)
        log.info(f"    SMOTE: {y_tr.sum()} → {y_res.sum()} positives")
        return X_res, y_res
    except Exception as e:
        log.warning(f"    SMOTE failed ({e}) — using original data")
        return X_tr, y_tr


def evaluate_fold(y_true, y_prob, threshold, fold_idx):
    """Compute all metrics for one fold at given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "fold":      fold_idx,
        "threshold": threshold,
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_prob),
        "pr_auc":    average_precision_score(y_true, y_prob),
        "brier":     brier_score_loss(y_true, y_prob),
    }


def print_cv_summary(name: str, results: list):
    keys = ["f1", "recall", "precision", "roc_auc", "pr_auc"]
    log.info(f"\n{'─'*60}")
    log.info(f"  {name}  —  {N_FOLDS}-Fold CV Summary")
    log.info(f"{'─'*60}")
    for k in keys:
        vals = [r[k] for r in results]
        log.info(f"  {k:<12}  {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    log.info(f"{'─'*60}")


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — DATA LOADING
# ════════════════════════════════════════════════════════════════════════════
def load_data() -> tuple:
    """
    Load the best available dataset in priority order:
      1. DataSet_Engineered_Cleaned.csv  (post-FE + zero-col cleanup)
      2. DataSet_Engineered.csv          (post-FE)
      3. DataSet_Cleaned.csv             (post-cleaning only)

    Returns X (DataFrame), y (Series), feature_names (list).
    """
    candidates = [
        "DataSet_Engineered_Cleaned.csv",
        "DataSet_Engineered.csv",
        "DataSet_Cleaned.csv",
    ]
    df = None
    for fname in candidates:
        if Path(fname).exists():
            log.info(f"Loading '{fname}' …")
            df = pd.read_csv(fname)
            log.info(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
            break

    if df is None:
        log.error("No dataset found. Run data_pipeline.py and/or feature_engineering_pipeline.py first.")
        sys.exit(1)

    # Drop index-like columns
    drop_candidates = ["Unnamed: 0", "index"]
    df.drop(columns=[c for c in drop_candidates if c in df.columns], inplace=True)

    # Sanitise column names (required by XGBoost / LightGBM)
    df = sanitize_col_names(df)

    # Drop columns that are 100% missing
    all_null = df.columns[df.isnull().mean() == 1.0].tolist()
    if all_null:
        df.drop(columns=all_null, inplace=True)
        log.info(f"  Dropped {len(all_null)} fully-null columns")

    if TARGET not in df.columns:
        log.error(f"Target column '{TARGET}' not found after sanitisation. "
                  "Check column names in the dataset.")
        sys.exit(1)

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])

    # Fill remaining NaNs with median (safe for tree models too)
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)  # catch any remaining (e.g. all-NaN columns after median)

    # Enforce float32 to reduce memory
    X = X.astype(np.float32)

    log.info(f"  Features  : {X.shape[1]:,}")
    log.info(f"  Positives : {y.sum()} ({y.mean()*100:.2f}%)")
    log.info(f"  scale_pos_weight: {compute_class_weight(y.values):.1f}")

    return X, y, X.columns.tolist()


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — LIGHTGBM
# ════════════════════════════════════════════════════════════════════════════
def get_lgbm_default_params(spw: float) -> dict:
    return {
        "objective":          "binary",
        "metric":             "average_precision",
        "n_estimators":       1000,
        "learning_rate":      0.03,
        "num_leaves":         63,
        "max_depth":          -1,
        "min_child_samples":  5,
        "subsample":          0.8,
        "colsample_bytree":   0.8,
        "reg_alpha":          0.1,
        "reg_lambda":         1.0,
        "scale_pos_weight":   spw,
        "random_state":       RANDOM_STATE,
        "n_jobs":             -1,
        "verbose":            -1,
    }


def tune_lgbm(X_tr, y_tr, X_val, y_val, spw: float, n_trials: int = OPTUNA_TRIALS) -> dict:
    """Optuna hyperparameter search for LightGBM."""
    if not OPTUNA_AVAILABLE:
        return get_lgbm_default_params(spw)

    def objective(trial):
        params = {
            "objective":         "binary",
            "metric":            "average_precision",
            "n_estimators":      trial.suggest_int("n_estimators", 300, 1500),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight":  spw,
            "random_state":      RANDOM_STATE,
            "n_jobs":            -1,
            "verbose":           -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        prob = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, prob)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best.update({"objective": "binary", "metric": "average_precision",
                 "scale_pos_weight": spw, "random_state": RANDOM_STATE,
                 "n_jobs": -1, "verbose": -1})
    log.info(f"    LGBM best PR-AUC: {study.best_value:.4f}")
    return best


def train_lgbm_cv(X: pd.DataFrame, y: pd.Series, skf: StratifiedKFold) -> dict:
    log.info("\n" + "═"*60)
    log.info("  [1/3] LIGHTGBM — Cross-Validation")
    log.info("═"*60)

    spw         = compute_class_weight(y.values)
    oof_probs   = np.zeros(len(y))
    fold_results = []
    best_params  = None

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        log.info(f"\n  Fold {fold}/{N_FOLDS} …")
        X_tr, X_val = X.iloc[tr_idx].values, X.iloc[val_idx].values
        y_tr, y_val = y.iloc[tr_idx].values, y.iloc[val_idx].values

        # Apply SMOTE only on training fold
        X_tr_s, y_tr_s = apply_smote(X_tr, y_tr)

        if fold == 1:
            log.info("    Optimising hyperparameters (Optuna) …")
            best_params = tune_lgbm(X_tr_s, y_tr_s, X_val, y_val, spw)

        model = lgb.LGBMClassifier(**best_params)
        model.fit(
            X_tr_s, y_tr_s,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        oof_probs[val_idx] = model.predict_proba(X_val)[:, 1]

        thresh_info = find_optimal_threshold(y_val, oof_probs[val_idx])
        res         = evaluate_fold(y_val, oof_probs[val_idx], thresh_info["threshold"], fold)
        fold_results.append(res)
        log.info(f"    F1={res['f1']:.4f}  Recall={res['recall']:.4f}  "
                 f"PR-AUC={res['pr_auc']:.4f}  ROC-AUC={res['roc_auc']:.4f}")

    print_cv_summary("LightGBM", fold_results)
    global_thresh = find_optimal_threshold(y.values, oof_probs)

    # Final model on full data
    log.info("\n  Training final LightGBM on full dataset …")
    X_full_s, y_full_s = apply_smote(X.values, y.values)
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_full_s, y_full_s)
    joblib.dump(final_model, MODEL_DIR / "lgbm_best.pkl")

    return {
        "name":        "LightGBM",
        "model":       final_model,
        "oof_probs":   oof_probs,
        "fold_results": fold_results,
        "threshold":   global_thresh["threshold"],
        "pr_auc":      global_thresh["pr_auc"],
        "roc_auc":     global_thresh["roc_auc"],
        "best_params": best_params,
        "feature_names": X.columns.tolist(),
    }


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — XGBOOST
# ════════════════════════════════════════════════════════════════════════════
def get_xgb_default_params(spw: float) -> dict:
    return {
        "objective":        "binary:logistic",
        "eval_metric":      "aucpr",
        "n_estimators":     1000,
        "learning_rate":    0.03,
        "max_depth":        6,
        "min_child_weight": 5,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "gamma":            0.1,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "scale_pos_weight": spw,
        "tree_method":      "hist",
        "random_state":     RANDOM_STATE,
        "n_jobs":           -1,
        "verbosity":        0,
    }


def tune_xgb(X_tr, y_tr, X_val, y_val, spw: float, n_trials: int = OPTUNA_TRIALS) -> dict:
    if not OPTUNA_AVAILABLE:
        return get_xgb_default_params(spw)

    def objective(trial):
        params = {
            "objective":           "binary:logistic",
            "eval_metric":         "aucpr",
            "n_estimators":        trial.suggest_int("n_estimators", 300, 1500),
            "learning_rate":       trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_depth":           trial.suggest_int("max_depth", 3, 10),
            "min_child_weight":    trial.suggest_int("min_child_weight", 1, 20),
            "subsample":           trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":    trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "gamma":               trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":           trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":          trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight":    spw,
            "tree_method":         "hist",
            "random_state":        RANDOM_STATE,
            "n_jobs":              -1,
            "verbosity":           0,
            # XGBoost >=2.0: early_stopping_rounds goes in the constructor, not fit()
            "early_stopping_rounds": 50,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        prob = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, prob)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best.update({
        "objective":             "binary:logistic",
        "eval_metric":           "aucpr",
        "scale_pos_weight":      spw,
        "tree_method":           "hist",
        "random_state":          RANDOM_STATE,
        "n_jobs":                -1,
        "verbosity":             0,
        "early_stopping_rounds": 50,   # XGBoost >=2.0: lives in constructor
    })
    log.info(f"    XGB best PR-AUC: {study.best_value:.4f}")
    return best


def train_xgb_cv(X: pd.DataFrame, y: pd.Series, skf: StratifiedKFold) -> dict:
    log.info("\n" + "═"*60)
    log.info("  [2/3] XGBOOST — Cross-Validation")
    log.info("═"*60)

    spw          = compute_class_weight(y.values)
    oof_probs    = np.zeros(len(y))
    fold_results = []
    best_params  = None

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        log.info(f"\n  Fold {fold}/{N_FOLDS} …")
        X_tr, X_val = X.iloc[tr_idx].values, X.iloc[val_idx].values
        y_tr, y_val = y.iloc[tr_idx].values, y.iloc[val_idx].values
        X_tr_s, y_tr_s = apply_smote(X_tr, y_tr)

        if fold == 1:
            log.info("    Optimising hyperparameters (Optuna) …")
            best_params = tune_xgb(X_tr_s, y_tr_s, X_val, y_val, spw)

        # best_params already contains early_stopping_rounds in the constructor
        model = xgb.XGBClassifier(**best_params)
        model.fit(
            X_tr_s, y_tr_s,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        oof_probs[val_idx] = model.predict_proba(X_val)[:, 1]

        thresh_info = find_optimal_threshold(y_val, oof_probs[val_idx])
        res         = evaluate_fold(y_val, oof_probs[val_idx], thresh_info["threshold"], fold)
        fold_results.append(res)
        log.info(f"    F1={res['f1']:.4f}  Recall={res['recall']:.4f}  "
                 f"PR-AUC={res['pr_auc']:.4f}  ROC-AUC={res['roc_auc']:.4f}")

    print_cv_summary("XGBoost", fold_results)
    global_thresh = find_optimal_threshold(y.values, oof_probs)

    log.info("\n  Training final XGBoost on full dataset …")
    X_full_s, y_full_s = apply_smote(X.values, y.values)
    # Final model: no eval_set, so remove early_stopping_rounds to avoid warning
    final_params = {k: v for k, v in best_params.items() if k != "early_stopping_rounds"}
    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(X_full_s, y_full_s, verbose=False)
    joblib.dump(final_model, MODEL_DIR / "xgb_best.pkl")

    return {
        "name":        "XGBoost",
        "model":       final_model,
        "oof_probs":   oof_probs,
        "fold_results": fold_results,
        "threshold":   global_thresh["threshold"],
        "pr_auc":      global_thresh["pr_auc"],
        "roc_auc":     global_thresh["roc_auc"],
        "best_params": best_params,
        "feature_names": X.columns.tolist(),
    }


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — CATBOOST
# ════════════════════════════════════════════════════════════════════════════
def get_catboost_default_params(spw: float) -> dict:
    return {
        "iterations":        1000,
        "learning_rate":     0.03,
        "depth":             6,
        "l2_leaf_reg":       3.0,
        "border_count":      128,
        "scale_pos_weight":  spw,
        "eval_metric":       "PRAUC",
        "random_seed":       RANDOM_STATE,
        "thread_count":      -1,
        "verbose":           False,
        "early_stopping_rounds": 50,
    }


def tune_catboost(X_tr, y_tr, X_val, y_val, spw: float, n_trials: int = OPTUNA_TRIALS) -> dict:
    if not OPTUNA_AVAILABLE:
        return get_catboost_default_params(spw)

    def objective(trial):
        params = {
            "iterations":       trial.suggest_int("iterations", 300, 1500),
            "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "depth":            trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg":      trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "border_count":     trial.suggest_categorical("border_count", [32, 64, 128, 254]),
            "scale_pos_weight": spw,
            "eval_metric":      "PRAUC",
            "random_seed":      RANDOM_STATE,
            "thread_count":     -1,
            "verbose":          False,
            "early_stopping_rounds": 50,
        }
        model = CatBoostClassifier(**params)
        eval_pool = Pool(X_val, y_val)
        model.fit(Pool(X_tr, y_tr), eval_set=eval_pool, verbose=False)
        prob = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, prob)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best.update({"eval_metric": "PRAUC", "random_seed": RANDOM_STATE,
                 "thread_count": -1, "verbose": False,
                 "early_stopping_rounds": 50, "scale_pos_weight": spw})
    log.info(f"    CatBoost best PR-AUC: {study.best_value:.4f}")
    return best


def train_catboost_cv(X: pd.DataFrame, y: pd.Series, skf: StratifiedKFold) -> dict:
    log.info("\n" + "═"*60)
    log.info("  [3/3] CATBOOST — Cross-Validation")
    log.info("═"*60)

    spw          = compute_class_weight(y.values)
    oof_probs    = np.zeros(len(y))
    fold_results = []
    best_params  = None

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        log.info(f"\n  Fold {fold}/{N_FOLDS} …")
        X_tr, X_val = X.iloc[tr_idx].values, X.iloc[val_idx].values
        y_tr, y_val = y.iloc[tr_idx].values, y.iloc[val_idx].values
        X_tr_s, y_tr_s = apply_smote(X_tr, y_tr)

        if fold == 1:
            log.info("    Optimising hyperparameters (Optuna) …")
            best_params = tune_catboost(X_tr_s, y_tr_s, X_val, y_val, spw)

        model = CatBoostClassifier(**best_params)
        model.fit(
            Pool(X_tr_s, y_tr_s),
            eval_set=Pool(X_val, y_val),
            verbose=False,
        )
        oof_probs[val_idx] = model.predict_proba(X_val)[:, 1]

        thresh_info = find_optimal_threshold(y_val, oof_probs[val_idx])
        res         = evaluate_fold(y_val, oof_probs[val_idx], thresh_info["threshold"], fold)
        fold_results.append(res)
        log.info(f"    F1={res['f1']:.4f}  Recall={res['recall']:.4f}  "
                 f"PR-AUC={res['pr_auc']:.4f}  ROC-AUC={res['roc_auc']:.4f}")

    print_cv_summary("CatBoost", fold_results)
    global_thresh = find_optimal_threshold(y.values, oof_probs)

    log.info("\n  Training final CatBoost on full dataset …")
    X_full_s, y_full_s = apply_smote(X.values, y.values)
    final_model = CatBoostClassifier(**best_params)
    final_model.fit(Pool(X_full_s, y_full_s), verbose=False)
    joblib.dump(final_model, MODEL_DIR / "catboost_best.pkl")

    return {
        "name":        "CatBoost",
        "model":       final_model,
        "oof_probs":   oof_probs,
        "fold_results": fold_results,
        "threshold":   global_thresh["threshold"],
        "pr_auc":      global_thresh["pr_auc"],
        "roc_auc":     global_thresh["roc_auc"],
        "best_params": best_params,
        "feature_names": X.columns.tolist(),
    }


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — STACKING ENSEMBLE
# ════════════════════════════════════════════════════════════════════════════
def build_stacking_ensemble(
    y: pd.Series,
    model_results: list,
) -> dict:
    """
    Stack the OOF predictions of the three base models using a
    Logistic Regression meta-learner with calibration.
    """
    log.info("\n" + "═"*60)
    log.info("  STACKING ENSEMBLE — Meta-Learner")
    log.info("═"*60)

    # Build meta-feature matrix from OOF probabilities
    meta_X = np.column_stack([r["oof_probs"] for r in model_results])
    # Optionally add averaged probability as a feature
    meta_X = np.column_stack([meta_X, meta_X.mean(axis=1)])

    y_arr = y.values
    scaler = StandardScaler()
    meta_X_scaled = scaler.fit_transform(meta_X)

    # Simple LR meta-learner with class weight
    spw  = compute_class_weight(y_arr)
    meta = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    meta.fit(meta_X_scaled, y_arr)
    ens_probs = meta.predict_proba(meta_X_scaled)[:, 1]

    thresh_info = find_optimal_threshold(y_arr, ens_probs)
    log.info(f"  Ensemble OOF PR-AUC : {thresh_info['pr_auc']:.4f}")
    log.info(f"  Ensemble OOF ROC-AUC: {thresh_info['roc_auc']:.4f}")
    log.info(f"  Optimal threshold   : {thresh_info['threshold']:.4f}")
    log.info(f"  F1 @ threshold      : {thresh_info['f1']:.4f}")
    log.info(f"  Recall @ threshold  : {thresh_info['recall']:.4f}")

    joblib.dump((meta, scaler), MODEL_DIR / "ensemble_meta.pkl")

    return {
        "name":       "Ensemble",
        "meta":       meta,
        "scaler":     scaler,
        "oof_probs":  ens_probs,
        "threshold":  thresh_info["threshold"],
        "pr_auc":     thresh_info["pr_auc"],
        "roc_auc":    thresh_info["roc_auc"],
        "f1":         thresh_info["f1"],
        "recall":     thresh_info["recall"],
    }


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — SHAP EXPLANATIONS
# ════════════════════════════════════════════════════════════════════════════
def generate_shap_plots(
    result: dict,
    X: pd.DataFrame,
    max_display: int = 30,
):
    """
    Generate SHAP beeswarm and bar plots for the best individual model.
    Uses a sample of 2,000 rows max for speed.
    """
    log.info("\n  Generating SHAP explanations …")
    model        = result["model"]
    feat_names   = result["feature_names"]
    model_name   = result["name"]

    # Sample for speed
    n_sample = min(2000, len(X))
    rng      = np.random.default_rng(RANDOM_STATE)
    idx      = rng.choice(len(X), size=n_sample, replace=False)
    X_sample = X.iloc[idx].values.astype(np.float32)

    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # For binary classification list output
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        # ── Beeswarm ──────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(sv, X_sample, feature_names=feat_names,
                          show=False, max_display=max_display,
                          plot_type="dot")
        plt.title(f"SHAP Beeswarm — {model_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        bswarm_path = PLOT_DIR / "shap_beeswarm.png"
        plt.savefig(bswarm_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"    Beeswarm → {bswarm_path}")

        # ── Bar chart ─────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(sv, X_sample, feature_names=feat_names,
                          show=False, max_display=max_display,
                          plot_type="bar")
        plt.title(f"SHAP Feature Importance — {model_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        bar_path = PLOT_DIR / "shap_bar.png"
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"    Bar chart → {bar_path}")

        # Save top-30 importance as CSV
        mean_shap = pd.DataFrame({
            "feature":       feat_names,
            "mean_abs_shap": np.abs(sv).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False).head(50)
        mean_shap["rank"] = range(1, len(mean_shap) + 1)
        mean_shap.to_csv(MODEL_DIR / "shap_top50_features.csv", index=False)
        log.info(f"    Top-50 features → {MODEL_DIR / 'shap_top50_features.csv'}")

    except Exception as e:
        log.warning(f"  SHAP generation failed: {e}")


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — EVALUATION PLOTS
# ════════════════════════════════════════════════════════════════════════════
def plot_pr_roc_curves(y: pd.Series, model_results: list, ensemble_result: dict):
    """Precision-Recall and ROC curves for all models + ensemble."""
    colours = {"LightGBM": "#3498db", "XGBoost": "#e74c3c",
               "CatBoost": "#2ecc71", "Ensemble": "#9b59b6"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Model Comparison — OOF Predictions", fontsize=15, fontweight="bold")

    y_arr = y.values
    all_results = model_results + [ensemble_result]

    for res in all_results:
        name  = res["name"]
        probs = res["oof_probs"]
        col   = colours.get(name, "#555555")
        lw    = 2.5 if name == "Ensemble" else 1.8

        # PR curve
        prec, rec, _ = precision_recall_curve(y_arr, probs)
        prauc         = average_precision_score(y_arr, probs)
        axes[0].plot(rec, prec, color=col, lw=lw,
                     label=f"{name}  (PR-AUC={prauc:.4f})")

        # ROC curve
        fpr, tpr, _ = roc_curve(y_arr, probs)
        rocauc       = roc_auc_score(y_arr, probs)
        axes[1].plot(fpr, tpr, color=col, lw=lw,
                     label=f"{name}  (ROC-AUC={rocauc:.4f})")

    # Baselines
    pos_rate = y_arr.mean()
    axes[0].axhline(pos_rate, color="gray", ls="--", lw=1.2, label=f"Baseline (={pos_rate:.3f})")
    axes[0].set_xlabel("Recall", fontsize=12)
    axes[0].set_ylabel("Precision", fontsize=12)
    axes[0].set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].plot([0, 1], [0, 1], color="gray", ls="--", lw=1.2, label="Random (AUC=0.50)")
    axes[1].set_xlabel("False Positive Rate", fontsize=12)
    axes[1].set_ylabel("True Positive Rate", fontsize=12)
    axes[1].set_title("ROC Curve", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = PLOT_DIR / "pr_roc_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"    PR/ROC curves → {path}")


def plot_confusion_matrices(y: pd.Series, model_results: list, ensemble_result: dict):
    """Confusion matrices at the optimal threshold for each model."""
    all_results = model_results + [ensemble_result]
    n           = len(all_results)
    fig, axes   = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.suptitle("Confusion Matrices @ Optimal Threshold", fontsize=14, fontweight="bold")
    y_arr = y.values

    for ax, res in zip(axes, all_results):
        thresh = res["threshold"]
        y_pred = (res["oof_probs"] >= thresh).astype(int)
        cm     = confusion_matrix(y_arr, y_pred)
        f1_val = f1_score(y_arr, y_pred, zero_division=0)
        rec_val= recall_score(y_arr, y_pred, zero_division=0)

        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Neg", "Pred Pos"])
        ax.set_yticklabels(["True Neg", "True Pos"])
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_title(f"{res['name']}\nF1={f1_val:.3f}  Recall={rec_val:.3f}\n"
                     f"Threshold={thresh:.3f}", fontsize=10)

    plt.tight_layout()
    path = PLOT_DIR / "confusion_matrices.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"    Confusion matrices → {path}")


def plot_calibration(y: pd.Series, model_results: list, ensemble_result: dict):
    """Probability calibration plot."""
    all_results = model_results + [ensemble_result]
    colours     = {"LightGBM": "#3498db", "XGBoost": "#e74c3c",
                   "CatBoost": "#2ecc71", "Ensemble": "#9b59b6"}
    y_arr       = y.values

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")

    for res in all_results:
        frac_pos, mean_pred = calibration_curve(y_arr, res["oof_probs"], n_bins=10)
        ax.plot(mean_pred, frac_pos, marker="o", lw=2,
                color=colours.get(res["name"], "#555"),
                label=res["name"])

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Curve", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = PLOT_DIR / "calibration_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"    Calibration curve → {path}")


def plot_threshold_analysis(y: pd.Series, best_result: dict):
    """F1 / Precision / Recall vs threshold for the best model."""
    y_arr  = y.values
    probs  = best_result["oof_probs"]
    name   = best_result["name"]
    opt_th = best_result["threshold"]

    thresholds = np.linspace(0.01, 0.99, 200)
    f1s, precs, recs = [], [], []
    for t in thresholds:
        y_pred = (probs >= t).astype(int)
        f1s.append(f1_score(y_arr, y_pred, zero_division=0))
        precs.append(precision_score(y_arr, y_pred, zero_division=0))
        recs.append(recall_score(y_arr, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, f1s,   color="#e74c3c", lw=2, label="F1")
    ax.plot(thresholds, precs, color="#3498db", lw=2, label="Precision")
    ax.plot(thresholds, recs,  color="#2ecc71", lw=2, label="Recall")
    ax.axvline(opt_th, color="black", ls="--", lw=1.5,
               label=f"Optimal threshold = {opt_th:.3f}")
    ax.fill_between(thresholds, 0, f1s, alpha=0.08, color="#e74c3c")
    ax.set_xlabel("Classification Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Threshold Analysis — {name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = PLOT_DIR / "threshold_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"    Threshold analysis → {path}")


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — INFERENCE PIPELINE
# ════════════════════════════════════════════════════════════════════════════
class MuleAccountInferencePipeline:
    """
    Production-ready inference pipeline for mule account detection.

    Usage
    ─────
    >>> pipe = MuleAccountInferencePipeline.load("models/inference_pipeline.pkl")
    >>> result = pipe.predict(df_new)
    >>> result["risk_label"]    # 'MULE' or 'LEGITIMATE'
    >>> result["mule_prob"]     # float probability
    >>> result["risk_tier"]     # 'CRITICAL' / 'HIGH' / 'MEDIUM' / 'LOW'
    >>> result["explanation"]   # top features driving the prediction

    Prediction schema (per row)
    ────────────────────────────
    mule_prob    : float [0, 1]   — probability of being a mule account
    risk_label   : str            — 'MULE' or 'LEGITIMATE'
    risk_tier    : str            — 'CRITICAL'(>0.75) / 'HIGH'(0.5-0.75) /
                                    'MEDIUM'(0.25-0.5) / 'LOW'(<0.25)
    flag_count   : int            — number of base models that flagged (≥threshold)
    """

    def __init__(
        self,
        ensemble_meta,
        ensemble_scaler,
        base_models: list,
        thresholds: dict,
        feature_names: list,
        model_names: list,
    ):
        self.ensemble_meta    = ensemble_meta
        self.ensemble_scaler  = ensemble_scaler
        self.base_models      = base_models
        self.thresholds       = thresholds    # {model_name: threshold}
        self.feature_names    = feature_names
        self.model_names      = model_names
        self._shap_explainer  = None

    # ── Internal helpers ──────────────────────────────────────────────────
    def _align_features(self, df: pd.DataFrame) -> np.ndarray:
        """Align input columns to the training feature list."""
        df = sanitize_col_names(df.copy())
        missing = [c for c in self.feature_names if c not in df.columns]
        extra   = [c for c in df.columns if c not in self.feature_names]
        if missing:
            for c in missing:
                df[c] = 0.0
        df = df[self.feature_names].fillna(0).astype(np.float32)
        return df.values

    def _base_probs(self, X_arr: np.ndarray) -> np.ndarray:
        """Get probability from each base model. Returns shape (n, n_models)."""
        probs = []
        for model in self.base_models:
            try:
                p = model.predict_proba(X_arr)[:, 1]
            except Exception:
                p = np.zeros(len(X_arr))
            probs.append(p)
        return np.column_stack(probs)

    @staticmethod
    def _prob_to_tier(prob: float) -> str:
        if prob >= 0.75:  return "CRITICAL"
        if prob >= 0.50:  return "HIGH"
        if prob >= 0.25:  return "MEDIUM"
        return "LOW"

    # ── Public API ────────────────────────────────────────────────────────
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return ensemble mule probability for each row."""
        X_arr      = self._align_features(df)
        base_p     = self._base_probs(X_arr)
        meta_input = np.column_stack([base_p, base_p.mean(axis=1)])
        meta_scaled= self.ensemble_scaler.transform(meta_input)
        return self.ensemble_meta.predict_proba(meta_scaled)[:, 1]

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return full prediction DataFrame with risk labels."""
        X_arr      = self._align_features(df)
        base_p     = self._base_probs(X_arr)
        meta_input = np.column_stack([base_p, base_p.mean(axis=1)])
        meta_scaled= self.ensemble_scaler.transform(meta_input)
        ens_prob   = self.ensemble_meta.predict_proba(meta_scaled)[:, 1]
        ens_thresh = self.thresholds.get("Ensemble", 0.5)

        # Per-base-model flags
        flag_matrix = np.column_stack([
            (base_p[:, i] >= self.thresholds.get(name, 0.5)).astype(int)
            for i, name in enumerate(self.model_names)
        ])

        result = pd.DataFrame({
            "mule_prob":  ens_prob,
            "risk_label": ["MULE" if p >= ens_thresh else "LEGITIMATE" for p in ens_prob],
            "risk_tier":  [self._prob_to_tier(p) for p in ens_prob],
            "flag_count": flag_matrix.sum(axis=1),
        })

        # Per-model probabilities
        for i, name in enumerate(self.model_names):
            result[f"prob_{name.lower()}"] = base_p[:, i]

        return result

    def explain(self, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Return SHAP-based top-N feature contributions for each row.
        Uses the first base model (LightGBM).
        """
        if self._shap_explainer is None:
            self._shap_explainer = shap.TreeExplainer(self.base_models[0])

        X_arr = self._align_features(df)
        sv    = self._shap_explainer.shap_values(X_arr)
        if isinstance(sv, list):
            sv = sv[1]

        rows = []
        for i in range(len(df)):
            contrib = pd.Series(sv[i], index=self.feature_names).abs().sort_values(ascending=False)
            top     = contrib.head(top_n)
            rows.append({
                "top_features":        top.index.tolist(),
                "shap_contributions":  sv[i][top.index.map(
                    {f: j for j, f in enumerate(self.feature_names)}
                ).values].tolist(),
            })
        return pd.DataFrame(rows)

    # ── Persistence ───────────────────────────────────────────────────────
    def save(self, path: str):
        joblib.dump(self, path)
        log.info(f"  Inference pipeline saved → {path}")

    @classmethod
    def load(cls, path: str) -> "MuleAccountInferencePipeline":
        pipe = joblib.load(path)
        log.info(f"  Inference pipeline loaded ← {path}")
        return pipe


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — MAIN ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════
def main():
    start_time = datetime.now()
    log.info("=" * 60)
    log.info("  MULE ACCOUNT DETECTION — MODEL TRAINING")
    log.info(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    # ── Step 1: Load data ─────────────────────────────────────────────────
    X, y, feature_names = load_data()

    # ── Step 2: Cross-validation setup ───────────────────────────────────
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ── Step 3: Train all three models ───────────────────────────────────
    lgbm_result = train_lgbm_cv(X, y, skf)
    xgb_result  = train_xgb_cv(X, y, skf)
    cb_result   = train_catboost_cv(X, y, skf)
    model_results = [lgbm_result, xgb_result, cb_result]

    # ── Step 4: Stacking ensemble ─────────────────────────────────────────
    ensemble_result = build_stacking_ensemble(y, model_results)

    # ── Step 5: Select best individual model by OOF PR-AUC ───────────────
    best_individual = max(model_results, key=lambda r: r["pr_auc"])
    log.info(f"\n  Best individual model: {best_individual['name']}  "
             f"(PR-AUC = {best_individual['pr_auc']:.4f})")
    joblib.dump(best_individual["model"], MODEL_DIR / "best_model.pkl")

    # ── Step 6: Generate plots ────────────────────────────────────────────
    log.info("\n  Generating evaluation plots …")
    plot_pr_roc_curves(y, model_results, ensemble_result)
    plot_confusion_matrices(y, model_results, ensemble_result)
    plot_calibration(y, model_results, ensemble_result)
    plot_threshold_analysis(y, best_individual)

    # ── Step 7: SHAP explanations ─────────────────────────────────────────
    generate_shap_plots(best_individual, X)

    # ── Step 8: Save thresholds & CV results ─────────────────────────────
    thresholds = {
        r["name"]: float(r["threshold"]) for r in model_results
    }
    thresholds["Ensemble"] = float(ensemble_result["threshold"])

    cv_results = {}
    for r in model_results:
        cv_results[r["name"]] = {
            "pr_auc":    r["pr_auc"],
            "roc_auc":   r["roc_auc"],
            "threshold": r["threshold"],
            "fold_f1":   [f["f1"] for f in r["fold_results"]],
            "fold_recall": [f["recall"] for f in r["fold_results"]],
            "fold_prauc": [f["pr_auc"] for f in r["fold_results"]],
            "mean_f1":   np.mean([f["f1"] for f in r["fold_results"]]),
            "mean_recall": np.mean([f["recall"] for f in r["fold_results"]]),
            "best_params": {
                k: (v if not isinstance(v, (np.integer, np.floating)) else float(v))
                for k, v in r["best_params"].items()
            },
        }
    cv_results["Ensemble"] = {
        "pr_auc":    ensemble_result["pr_auc"],
        "roc_auc":   ensemble_result["roc_auc"],
        "f1":        ensemble_result["f1"],
        "recall":    ensemble_result["recall"],
        "threshold": ensemble_result["threshold"],
    }

    with open(MODEL_DIR / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    with open(MODEL_DIR / "cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=2)
    with open(MODEL_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    log.info(f"  Thresholds   → {MODEL_DIR}/thresholds.json")
    log.info(f"  CV results   → {MODEL_DIR}/cv_results.json")
    log.info(f"  Feature list → {MODEL_DIR}/feature_names.json")

    # ── Step 9: Build & save inference pipeline ───────────────────────────
    log.info("\n  Building inference pipeline …")
    inference_pipe = MuleAccountInferencePipeline(
        ensemble_meta    = ensemble_result["meta"],
        ensemble_scaler  = ensemble_result["scaler"],
        base_models      = [r["model"] for r in model_results],
        thresholds       = thresholds,
        feature_names    = feature_names,
        model_names      = [r["name"] for r in model_results],
    )
    inference_pipe.save(str(MODEL_DIR / "inference_pipeline.pkl"))

    # ── Step 10: Final summary ────────────────────────────────────────────
    elapsed = (datetime.now() - start_time).total_seconds()
    log.info("\n" + "═"*60)
    log.info("  FINAL RESULTS SUMMARY")
    log.info("═"*60)
    header = f"  {'Model':<15} {'PR-AUC':>8} {'ROC-AUC':>9} {'Threshold':>11} {'F1':>7} {'Recall':>8}"
    log.info(header)
    log.info("  " + "─"*58)
    for r in model_results:
        mean_f1  = np.mean([f["f1"] for f in r["fold_results"]])
        mean_rec = np.mean([f["recall"] for f in r["fold_results"]])
        log.info(f"  {r['name']:<15} {r['pr_auc']:>8.4f} {r['roc_auc']:>9.4f} "
                 f"{r['threshold']:>11.4f} {mean_f1:>7.4f} {mean_rec:>8.4f}")
    log.info("  " + "─"*58)
    log.info(f"  {'Ensemble':<15} {ensemble_result['pr_auc']:>8.4f} "
             f"{ensemble_result['roc_auc']:>9.4f} "
             f"{ensemble_result['threshold']:>11.4f} "
             f"{ensemble_result['f1']:>7.4f} "
             f"{ensemble_result['recall']:>8.4f}")
    log.info("═"*60)
    log.info(f"  Best model: {best_individual['name']}  (PR-AUC = {best_individual['pr_auc']:.4f})")
    log.info(f"  Elapsed   : {elapsed/60:.1f} minutes")
    log.info("═"*60)

    log.info("\n  Output files:")
    for f in sorted(MODEL_DIR.iterdir()):
        log.info(f"    {f}")
    for f in sorted(PLOT_DIR.iterdir()):
        log.info(f"    {f}")

    log.info("\n✅  Training complete.\n")

    # ── Quick inference demo ──────────────────────────────────────────────
    log.info("  Demo: running inference on 5 sample rows …")
    sample_df = X.sample(5, random_state=RANDOM_STATE)
    preds     = inference_pipe.predict(sample_df)
    log.info("\n" + preds[["mule_prob", "risk_label", "risk_tier", "flag_count"]].to_string())


if __name__ == "__main__":
    main()
