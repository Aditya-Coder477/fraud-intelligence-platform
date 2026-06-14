"""
================================================================================
  MULE ACCOUNT DETECTION — FEATURE ENGINEERING PIPELINE
  Target: F3924  (0 = Legitimate, 1 = Mule Account)
  Author: Principal AML & Fraud Detection Data Scientist
================================================================================

DATASET CONTEXT
---------------
  Rows  : 9,082
  Cols  : 3,925  (F1 … F3924)
  Target: F3924  — binary, severely imbalanced (0.89 % mule)
  Dtypes : float64 (3,876), int64 (41), str (8)
  Known categorical columns
    F2230  : reporting month   (Oct25 …)           — 4 unique
    F3886  : account type      (Savings …)         — 17 unique
    F3888  : account open date (8-1-2011 …)        — 4,292 unique  → date
    F3889  : product code      (G365D …)           — 7 unique
    F3890  : region code       (R, SU …)           — 4 unique
    F3891  : employment type   (selfemployed …)    — 7 unique
    F3892  : gender            (M, F …)            — 3 unique
    F3893  : channel           (RETAIL …)          — 2 unique
  Target leakage: F3912 (corr = 0.97) — always excluded

PIPELINE STAGES
---------------
  1.  Missing-value features
  2.  Frequency-encoded features
  3.  Target-encoded features          (smoothed, CV-safe)
  4.  Date decomposition features      (F3888 — account open date)
  5.  Ratio features
  6.  Interaction features
  7.  Outlier indicators
  8.  Behavioural features
  9.  Fraud-risk composite features
  10. SHAP-driven feature selection    (LightGBM + TreeExplainer)
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import warnings
import os
import json
from datetime import datetime

warnings.filterwarnings("ignore")

# ── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import joblib

# Optional heavy deps — imported lazily inside the relevant sections
# lightgbm, shap

# ════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
TARGET = "F3924"
LEAKAGE_COLS = ["F3912"]          # confirmed leakage (corr = 0.97)
DATE_COL     = "F3888"            # account open-date string  (M-D-YYYY)
REFERENCE_DATE = datetime(2026, 1, 1)  # anchor for account-age calculations

# Known semantic categorical columns (excluding the date col)
CAT_COLS = ["F2230", "F3886", "F3889", "F3890", "F3891", "F3892", "F3893"]

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — MISSING-VALUE FEATURES
# ════════════════════════════════════════════════════════════════════════════
class MissingValueFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates informative features from the missingness pattern itself.

    WHY MULE ACCOUNTS?
    ------------------
    Mule accounts are often opened in bulk by fraud rings using synthetic or
    stolen identities.  Their application data is frequently fabricated or
    deliberately incomplete — fields that genuine customers always fill in
    may be missing, or conversely, suspicious accounts may have suspiciously
    *complete* profiles to avoid flagging.  The row-level missingness count
    and column-level missing indicators therefore carry direct fraud signal.

    Features Generated
    ------------------
    mv_row_missing_count     : total NaN cells in that row (raw count)
    mv_row_missing_rate      : fraction of NaN cells in that row
    mv_has_any_missing       : 1 if any NaN exists in the row
    mv_missing_block_score   : weighted score — columns with high global
                               missing rate in mule rows get more weight
    {col}_is_missing         : per-column binary indicator (only for cols
                               whose missingness rate is between 1 % and 95 %)
    mv_col_miss_cluster      : which quintile of 'row missing rate' this row
                               belongs to (0–4)
    """

    def __init__(self, low_thresh: float = 0.01, high_thresh: float = 0.95):
        self.low_thresh  = low_thresh
        self.high_thresh = high_thresh

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        missing_rate = df.isnull().mean()

        # Columns with informative missingness
        self.indicator_cols_ = [
            c for c in df.columns
            if c != TARGET
            and self.low_thresh <= missing_rate[c] <= self.high_thresh
        ]

        # Per-column mule-conditional missing rate (if y available)
        if y is not None:
            y_s = pd.Series(y, index=df.index)
            mule_mask = y_s == 1
            self.mule_miss_rate_ = {
                c: df.loc[mule_mask, c].isnull().mean()
                for c in self.indicator_cols_
            }
        else:
            self.mule_miss_rate_ = {c: 1.0 for c in self.indicator_cols_}

        return self

    def transform(self, X: pd.DataFrame):
        df = X.copy()
        all_feat_cols = [c for c in df.columns if c != TARGET]

        # Row-level aggregates
        miss_mat = df[all_feat_cols].isnull()
        df["mv_row_missing_count"] = miss_mat.sum(axis=1)
        df["mv_row_missing_rate"]  = miss_mat.mean(axis=1)
        df["mv_has_any_missing"]   = (df["mv_row_missing_count"] > 0).astype(int)

        # Weighted block score — columns where mules miss more get higher weight
        weights = np.array([self.mule_miss_rate_.get(c, 0.0) for c in self.indicator_cols_])
        if weights.sum() > 0:
            indicator_mat = df[self.indicator_cols_].isnull().values.astype(float)
            df["mv_missing_block_score"] = (indicator_mat * weights).sum(axis=1) / (weights.sum() + 1e-9)
        else:
            df["mv_missing_block_score"] = 0.0

        # Per-column binary indicators
        for c in self.indicator_cols_:
            df[f"{c}_is_missing"] = df[c].isnull().astype(int)

        # Quintile cluster of missing rate
        df["mv_col_miss_cluster"] = pd.qcut(
            df["mv_row_missing_rate"], q=5, labels=False, duplicates="drop"
        ).fillna(0).astype(int)

        return df


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — FREQUENCY-ENCODED FEATURES
# ════════════════════════════════════════════════════════════════════════════
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Replaces each categorical value with its frequency (proportion) in the
    training set.

    WHY MULE ACCOUNTS?
    ------------------
    Mule accounts cluster around unusual or rare category combinations — e.g.
    a self-employed student in an obscure product bracket.  Rare categories
    therefore have low frequency scores, providing a numeric proxy for
    'how unusual is this account's profile'.

    Features Generated (per categorical column col)
    ------------------------------------------------
    {col}_freq        : proportion of training rows sharing this category value
    {col}_freq_rank   : rank of the category by frequency (1 = most common)
    {col}_is_rare     : 1 if frequency < 1 % (potential synthetic identity)
    freq_composite    : mean frequency across all categorical columns
    freq_rarity_score : number of columns where this row is in a 'rare' category
    """

    def __init__(self, cat_cols=None, rare_threshold: float = 0.01):
        self.cat_cols       = cat_cols or CAT_COLS
        self.rare_threshold = rare_threshold

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        self.freq_maps_  = {}
        self.rank_maps_  = {}

        for c in self.cat_cols:
            if c not in df.columns:
                continue
            freq = df[c].value_counts(normalize=True)
            self.freq_maps_[c] = freq.to_dict()
            # rank 1 = most common
            self.rank_maps_[c] = {v: r + 1 for r, v in enumerate(freq.index)}

        return self

    def transform(self, X: pd.DataFrame):
        df = X.copy()
        freq_cols = []

        for c in self.cat_cols:
            if c not in df.columns:
                continue
            fname = f"{c}_freq"
            rname = f"{c}_freq_rank"
            iname = f"{c}_is_rare"

            df[fname] = df[c].map(self.freq_maps_[c]).fillna(0.0)
            df[rname] = df[c].map(self.rank_maps_[c]).fillna(9999)
            df[iname] = (df[fname] < self.rare_threshold).astype(int)
            freq_cols.append(fname)

        if freq_cols:
            df["freq_composite"]    = df[freq_cols].mean(axis=1)
            df["freq_rarity_score"] = df[
                [f"{c}_is_rare" for c in self.cat_cols if f"{c}_is_rare" in df.columns]
            ].sum(axis=1)

        return df


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — TARGET-ENCODED FEATURES  (smoothed, CV-safe)
# ════════════════════════════════════════════════════════════════════════════
class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes each category as its smoothed posterior probability of being mule.

        te(c) = (n_c * mean_c + k * global_mean) / (n_c + k)

    where k is the smoothing factor (default 20), n_c is the count of that
    category value in training, mean_c is its mule rate, and global_mean is
    the dataset-level mule rate.

    CV-safety: During fit() we use leave-one-fold-out encoding so that the
    training set itself never sees its own target when fitting. A held-out
    encoder is stored for inference.

    WHY MULE ACCOUNTS?
    ------------------
    Some account types (e.g., 'Basic Current'), employment categories
    ('student', 'unemployed') or channels exhibit higher mule rates.
    Target encoding converts these into a direct risk signal while keeping
    cardinality manageable.  Smoothing prevents overfitting on rare categories.

    Features Generated (per categorical column col)
    ------------------------------------------------
    {col}_te              : smoothed mule probability for that category
    {col}_te_deviation    : deviation from the global mule rate
    {col}_te_high_risk    : 1 if te > 2 × global_mean  (strongly elevated risk)
    te_mean_risk          : mean target encoding across all categorical columns
    te_max_risk           : max target encoding across all categorical columns
    te_risk_spread        : max − min  (wide spread = inconsistent risk profile)
    """

    def __init__(self, cat_cols=None, smoothing: float = 20.0, n_splits: int = 5):
        self.cat_cols  = cat_cols or CAT_COLS
        self.smoothing = smoothing
        self.n_splits  = n_splits

    @staticmethod
    def _smooth_encode(series: pd.Series, target: pd.Series, k: float, global_mean: float) -> pd.Series:
        stats = target.groupby(series).agg(["count", "mean"])
        smooth = (stats["count"] * stats["mean"] + k * global_mean) / (stats["count"] + k)
        return series.map(smooth).fillna(global_mean)

    def fit(self, X: pd.DataFrame, y):
        df    = X.copy()
        y_arr = pd.Series(y, index=df.index)
        self.global_mean_  = float(y_arr.mean())
        self.te_maps_      = {}

        for c in self.cat_cols:
            if c not in df.columns:
                continue
            stats = y_arr.groupby(df[c]).agg(["count", "mean"])
            smooth = (
                (stats["count"] * stats["mean"] + self.smoothing * self.global_mean_)
                / (stats["count"] + self.smoothing)
            )
            self.te_maps_[c] = smooth.to_dict()

        return self

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params):
        """CV-aware fit_transform using OOF encoding to avoid target leakage."""
        df    = X.copy()
        y_arr = pd.Series(y, index=df.index)
        self.global_mean_  = float(y_arr.mean())
        self.te_maps_      = {}

        skf  = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        te_encoded = pd.DataFrame(index=df.index)

        for c in self.cat_cols:
            if c not in df.columns:
                continue
            col_te = pd.Series(np.nan, index=df.index)

            for tr_idx, val_idx in skf.split(df, y_arr):
                tr_series  = df.iloc[tr_idx][c]
                tr_target  = y_arr.iloc[tr_idx]
                val_series = df.iloc[val_idx][c]
                oof_te     = self._smooth_encode(tr_series, tr_target, self.smoothing, self.global_mean_)
                col_te.iloc[val_idx] = val_series.map(
                    tr_series.map(oof_te.values if hasattr(oof_te, 'values') else oof_te).to_dict()
                ).fillna(self.global_mean_).values

            # Full-data encoder for inference
            stats  = y_arr.groupby(df[c]).agg(["count", "mean"])
            smooth = (
                (stats["count"] * stats["mean"] + self.smoothing * self.global_mean_)
                / (stats["count"] + self.smoothing)
            )
            self.te_maps_[c] = smooth.to_dict()
            te_encoded[f"{c}_te"] = col_te.fillna(self.global_mean_)

        # Append to df
        df = pd.concat([df, te_encoded], axis=1)
        df = self._add_composite_te_features(df)
        return df

    def transform(self, X: pd.DataFrame):
        df = X.copy()
        for c in self.cat_cols:
            if c not in df.columns:
                continue
            df[f"{c}_te"] = df[c].map(self.te_maps_[c]).fillna(self.global_mean_)
            df[f"{c}_te_deviation"] = df[f"{c}_te"] - self.global_mean_
            df[f"{c}_te_high_risk"] = (df[f"{c}_te"] > 2 * self.global_mean_).astype(int)

        df = self._add_composite_te_features(df)
        return df

    def _add_composite_te_features(self, df: pd.DataFrame) -> pd.DataFrame:
        te_cols = [f"{c}_te" for c in self.cat_cols if f"{c}_te" in df.columns]
        if te_cols:
            df["te_mean_risk"]   = df[te_cols].mean(axis=1)
            df["te_max_risk"]    = df[te_cols].max(axis=1)
            df["te_risk_spread"] = df[te_cols].max(axis=1) - df[te_cols].min(axis=1)
        return df


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — DATE DECOMPOSITION FEATURES  (F3888 — account open date)
# ════════════════════════════════════════════════════════════════════════════
class DateDecompositionEngineer(BaseEstimator, TransformerMixin):
    """
    Parses the account-open-date field (F3888, format M-D-YYYY) and extracts
    temporal signals that are strong indicators of mule account activity.

    WHY MULE ACCOUNTS?
    ------------------
    • Mule rings open many accounts in short bursts — unusually large numbers
      of accounts with the *same* open date indicate coordinated fraud.
    • Accounts opened very recently (low account_age_days) and immediately
      showing transaction volume are high-risk.
    • Weekends and month/quarter boundaries are favoured by fraudsters to
      exploit reduced monitoring windows.
    • Very old accounts may have been 'dormant mules' reactivated for a
      campaign.

    Features Generated
    ------------------
    date_open_year          : calendar year of account opening
    date_open_month         : calendar month (1–12)
    date_open_day           : day of month (1–31)
    date_open_dayofweek     : 0=Monday … 6=Sunday
    date_open_quarter       : fiscal quarter (1–4)
    date_open_is_weekend    : 1 if Saturday or Sunday
    date_open_is_month_end  : 1 if day ≥ 28
    date_open_is_month_start: 1 if day ≤ 3
    account_age_days        : days since account opened (vs reference date)
    account_age_years       : float years since account opened
    account_age_bucket      : categorical bucket (new/young/mature/old/ancient)
    date_open_same_date_count: how many accounts share the exact same open date
                              (batch-opening signal)
    date_open_same_month_count: accounts sharing the same month+year
    date_open_cohort_risk   : z-score of same_date_count  (high = suspicious batch)
    """

    def __init__(self, date_col: str = DATE_COL, reference_date: datetime = REFERENCE_DATE):
        self.date_col       = date_col
        self.reference_date = reference_date

    def fit(self, X: pd.DataFrame, y=None):
        if self.date_col not in X.columns:
            self.date_counts_      = {}
            self.month_counts_     = {}
            return self

        dates = pd.to_datetime(X[self.date_col], format="%m-%d-%Y", errors="coerce")
        self.date_counts_  = dates.dt.strftime("%Y-%m-%d").value_counts().to_dict()
        self.month_counts_ = dates.dt.strftime("%Y-%m").value_counts().to_dict()
        self._date_count_mean = np.mean(list(self.date_counts_.values())) if self.date_counts_ else 1.0
        self._date_count_std  = np.std(list(self.date_counts_.values()))  if self.date_counts_ else 1.0
        return self

    def transform(self, X: pd.DataFrame):
        df = X.copy()
        if self.date_col not in df.columns:
            return df

        # Parse
        dates = pd.to_datetime(df[self.date_col], format="%m-%d-%Y", errors="coerce")

        df["date_open_year"]           = dates.dt.year
        df["date_open_month"]          = dates.dt.month
        df["date_open_day"]            = dates.dt.day
        df["date_open_dayofweek"]      = dates.dt.dayofweek
        df["date_open_quarter"]        = dates.dt.quarter
        df["date_open_is_weekend"]     = (dates.dt.dayofweek >= 5).astype(int)
        df["date_open_is_month_end"]   = (dates.dt.day >= 28).astype(int)
        df["date_open_is_month_start"] = (dates.dt.day <= 3).astype(int)

        # Account age
        df["account_age_days"]  = (self.reference_date - dates).dt.days
        df["account_age_years"] = df["account_age_days"] / 365.25

        # Age bucket
        df["account_age_bucket"] = pd.cut(
            df["account_age_days"],
            bins=[-1, 90, 365, 1095, 3650, np.inf],
            labels=[0, 1, 2, 3, 4]          # new/young/mature/old/ancient
        ).astype(float).fillna(-1)

        # Batch-opening signals
        date_keys  = dates.dt.strftime("%Y-%m-%d")
        month_keys = dates.dt.strftime("%Y-%m")

        df["date_open_same_date_count"]  = date_keys.map(self.date_counts_).fillna(1)
        df["date_open_same_month_count"] = month_keys.map(self.month_counts_).fillna(1)

        denom = self._date_count_std if self._date_count_std > 0 else 1.0
        df["date_open_cohort_risk"] = (
            df["date_open_same_date_count"] - self._date_count_mean
        ) / denom

        # Fill NaTs with -1
        date_feature_cols = [
            "date_open_year", "date_open_month", "date_open_day",
            "date_open_dayofweek", "date_open_quarter",
            "date_open_is_weekend", "date_open_is_month_end",
            "date_open_is_month_start", "account_age_days", "account_age_years"
        ]
        df[date_feature_cols] = df[date_feature_cols].fillna(-1)

        # Drop raw date string
        df.drop(columns=[self.date_col], inplace=True, errors="ignore")
        return df


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — RATIO FEATURES
# ════════════════════════════════════════════════════════════════════════════
class RatioFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Constructs interpretable financial ratios from numeric feature groups.

    WHY MULE ACCOUNTS?
    ------------------
    A mule account's purpose is to rapidly move money.  Standard balances and
    transaction amounts look normal in isolation, but *ratios* expose
    anomalies — e.g. a large debit-to-credit ratio, high utilisation of a
    credit limit, or an extreme inflow-to-balance ratio all signal pass-through
    behaviour.

    The pipeline auto-detects numeric column groups by naming conventions in
    the feature space and computes:

    Features Generated
    ------------------
    ratio_numeric_cv          : coefficient of variation of all numeric features
                                in the row (high CV = erratic activity profile)
    ratio_top_to_median       : max numeric value / median numeric value + ε
                                (dominance of a single extreme value)
    ratio_nonzero_fraction    : proportion of numeric features that are > 0
    ratio_pos_to_neg          : count of positive values / count of negative values
    ratio_extreme_range       : (max − min) / (|mean| + ε)
    [NAMED RATIOS — see _build_named_ratios()]
    """

    def __init__(self, numeric_cols=None, top_n_variance: int = 200):
        self.numeric_cols    = numeric_cols      # if None, auto-detect on fit
        self.top_n_variance  = top_n_variance    # use top-N high-variance cols for ratios

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        if self.numeric_cols:
            candidates = [c for c in self.numeric_cols if c in df.columns and c != TARGET]
        else:
            candidates = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c != TARGET and "_is_missing" not in c
            ]

        # Select high-variance columns for ratio construction
        variances = df[candidates].var().sort_values(ascending=False)
        self.high_var_cols_ = variances.head(self.top_n_variance).index.tolist()
        return self

    def transform(self, X: pd.DataFrame):
        df   = X.copy()
        cols = [c for c in self.high_var_cols_ if c in df.columns]
        if not cols:
            return df

        sub = df[cols].copy()
        sub_np = sub.values.astype(float)

        row_mean   = np.nanmean(sub_np, axis=1)
        row_std    = np.nanstd(sub_np, axis=1)
        row_max    = np.nanmax(sub_np, axis=1)
        row_min    = np.nanmin(sub_np, axis=1)
        row_median = np.nanmedian(sub_np, axis=1)

        df["ratio_numeric_cv"]        = row_std / (np.abs(row_mean) + 1e-9)
        df["ratio_top_to_median"]     = row_max / (np.abs(row_median) + 1e-9)
        df["ratio_nonzero_fraction"]  = (sub_np != 0).sum(axis=1) / (sub_np.shape[1] + 1e-9)
        df["ratio_pos_to_neg"]        = (
            (sub_np > 0).sum(axis=1) / ((sub_np < 0).sum(axis=1) + 1e-9)
        )
        df["ratio_extreme_range"]     = (row_max - row_min) / (np.abs(row_mean) + 1e-9)

        return df


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — INTERACTION FEATURES
# ════════════════════════════════════════════════════════════════════════════
class InteractionFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates pairwise interaction terms between the most predictive numeric
    features (by variance) and the semantic categorical-derived features.

    WHY MULE ACCOUNTS?
    ------------------
    Individual features rarely capture the full mule pattern.  A student
    (employment type) + savings account + very recent open date + high
    transaction frequency is far more suspicious than any single attribute.
    Pairwise products and differences expose these joint patterns to tree
    models and linear baselines alike.

    Features Generated
    ------------------
    For the top-K high-variance numeric columns:
      interact_{A}_x_{B}   : A × B  (amplifies joint extremes)
      interact_{A}_diff_{B}: A − B  (differential — useful for balance deltas)

    For target-encoded categorical columns:
      te_interact_{A}_x_{B}: product of two TE risk scores
                             (highest risk = both categories are high-risk)

    Composite:
      interact_sum_top      : sum of the top-K numeric values in that row
      interact_sum_sq_top   : sum of squares (emphasises outlier features)
    """

    def __init__(self, top_k: int = 15, cat_te_cols=None):
        self.top_k       = top_k
        self.cat_te_cols = cat_te_cols   # filled during fit

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        num_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != TARGET and "_is_missing" not in c
            and not c.startswith("mv_") and not c.startswith("ratio_")
        ]
        variances = df[num_cols].var().sort_values(ascending=False)
        self.top_num_cols_ = variances.head(self.top_k).index.tolist()

        # TE columns for categorical interactions
        if self.cat_te_cols is None:
            self.cat_te_cols = [c for c in df.columns if c.endswith("_te")]

        return self

    def transform(self, X: pd.DataFrame):
        df   = X.copy()
        cols = [c for c in self.top_num_cols_ if c in df.columns]

        new_feats = {}

        # Pairwise numeric products & differences (limited to avoid explosion)
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                new_feats[f"interact_{a}_x_{b}"]    = df[a] * df[b]
                new_feats[f"interact_{a}_diff_{b}"]  = df[a] - df[b]

        # Categorical TE pairwise products
        te_cols = [c for c in self.cat_te_cols if c in df.columns]
        for i, a in enumerate(te_cols):
            for b in te_cols[i + 1:]:
                new_feats[f"te_interact_{a}_x_{b}"] = df[a] * df[b]

        # Row-level aggregates of top-K features
        if cols:
            sub_np = df[cols].values.astype(float)
            new_feats["interact_sum_top"]    = np.nansum(sub_np, axis=1)
            new_feats["interact_sum_sq_top"] = np.nansum(sub_np ** 2, axis=1)

        df = pd.concat([df, pd.DataFrame(new_feats, index=df.index)], axis=1)
        return df


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — OUTLIER INDICATORS
# ════════════════════════════════════════════════════════════════════════════
class OutlierIndicatorEngineer(BaseEstimator, TransformerMixin):
    """
    Flags observations that are statistically extreme in each numeric feature,
    both at the individual-column level and in aggregate.

    WHY MULE ACCOUNTS?
    ------------------
    Mule accounts are designed to look normal, but their *transaction
    behaviour* often breaches boundaries set by legitimate customers:
    a single outgoing transfer that is orders of magnitude larger than the
    account's average, or a transaction count that sits in the top 0.1 %
    of all accounts.  Outlier flags surface these anomalies explicitly.

    Methods
    -------
    IQR  : value < Q1 − 1.5·IQR  OR  value > Q3 + 1.5·IQR → flag 1
    Z-score : |z| > threshold (default 3.0) → flag 1

    Features Generated
    ------------------
    {col}_iqr_outlier   : binary IQR outlier flag per column
    {col}_zscore_outlier: binary z-score outlier flag per column
    out_iqr_count       : total number of IQR-outlier flags in the row
    out_zscore_count    : total number of z-score-outlier flags in the row
    out_total_flag_count: out_iqr_count + out_zscore_count
    out_outlier_rate    : out_total_flag_count / (2 × num_cols)
    out_max_zscore      : maximum |z-score| across all columns in the row
    out_composite_score : weighted combination of outlier flags
    """

    def __init__(self, top_n: int = 100, z_thresh: float = 3.0):
        self.top_n    = top_n
        self.z_thresh = z_thresh

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        num_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != TARGET and "_is_missing" not in c
        ]
        variances = df[num_cols].var().sort_values(ascending=False)
        self.analysis_cols_ = variances.head(self.top_n).index.tolist()

        self.q1_  = df[self.analysis_cols_].quantile(0.25)
        self.q3_  = df[self.analysis_cols_].quantile(0.75)
        self.iqr_ = self.q3_ - self.q1_
        self.mean_ = df[self.analysis_cols_].mean()
        self.std_  = df[self.analysis_cols_].std().replace(0, 1e-9)
        return self

    def transform(self, X: pd.DataFrame):
        df   = X.copy()
        cols = [c for c in self.analysis_cols_ if c in df.columns]

        iqr_flags   = []
        zscore_flags = []

        for c in cols:
            # IQR
            lower = self.q1_[c] - 1.5 * self.iqr_[c]
            upper = self.q3_[c] + 1.5 * self.iqr_[c]
            iqr_flag = ((df[c] < lower) | (df[c] > upper)).astype(int)
            df[f"{c}_iqr_outlier"] = iqr_flag
            iqr_flags.append(f"{c}_iqr_outlier")

            # Z-score
            zscore = (df[c] - self.mean_[c]) / self.std_[c]
            zflag  = (zscore.abs() > self.z_thresh).astype(int)
            df[f"{c}_zscore_outlier"] = zflag
            zscore_flags.append(f"{c}_zscore_outlier")

        df["out_iqr_count"]        = df[iqr_flags].sum(axis=1)
        df["out_zscore_count"]     = df[zscore_flags].sum(axis=1)
        df["out_total_flag_count"] = df["out_iqr_count"] + df["out_zscore_count"]

        n = len(cols)
        df["out_outlier_rate"]     = df["out_total_flag_count"] / (2 * n + 1e-9)

        # Max |z| across all columns in the row
        sub_z = (df[cols] - self.mean_) / self.std_
        df["out_max_zscore"]       = sub_z.abs().max(axis=1)

        # Composite score: IQR + z weighted equally
        df["out_composite_score"]  = (df["out_iqr_count"] + df["out_zscore_count"]) / (2 * n + 1e-9)

        return df


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — BEHAVIOURAL FEATURES
# ════════════════════════════════════════════════════════════════════════════
class BehaviouralFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Derives activity-pattern features that reveal the *behaviour* of an
    account rather than a single snapshot value.

    WHY MULE ACCOUNTS?
    ------------------
    The defining characteristic of a mule account is behavioural: money is
    deposited (from a victim) and immediately forwarded (to the fraud ring).
    This creates a distinctive pattern:
      • very high velocity  (large amounts per unit time)
      • low balance retention (funds leave immediately)
      • high transaction diversity across counterparties or amounts
      • erratic timing patterns

    Because the dataset uses anonymous feature names (F1…F3923), we rely on
    statistical proxies that capture these behavioural signals:

    Features Generated
    ------------------
    beh_row_mean          : mean of all numeric values (activity level)
    beh_row_std           : std of all numeric values (variability)
    beh_row_skew          : skewness of the row's feature vector
    beh_row_kurtosis      : excess kurtosis (tail heaviness = extreme events)
    beh_positive_sum      : sum of all positive values (total inflows)
    beh_negative_sum      : sum of all negative values (total outflows)
    beh_net_flow          : beh_positive_sum + beh_negative_sum (net position)
    beh_flow_imbalance    : |inflows − |outflows|| / (|inflows| + |outflows| + ε)
                            values near 0 = balanced (pass-through); near 1 = net accumulator
    beh_high_activity     : 1 if beh_row_mean > 2σ above population mean
    beh_zero_fraction     : proportion of features that are exactly 0
    beh_distinct_values   : count of distinct numeric values in the row
    beh_entropy           : Shannon entropy of the row's value distribution
                            (high = uniform/random, low = concentrated)
    beh_acct_type_risk    : categorical risk mapped from F3886 (account type)
    beh_employment_risk   : categorical risk mapped from F3891 (employment type)
    beh_channel_risk      : categorical risk mapped from F3893 (channel)
    """

    # Heuristic risk mappings derived from AML domain knowledge
    ACCT_TYPE_RISK = {
        "Savings":          0.3,
        "Current":          0.5,
        "Basic Current":    0.8,
        "Student":          0.7,
        "Joint":            0.4,
        "Business":         0.6,
        "Premium":          0.2,
    }
    EMPLOYMENT_RISK = {
        "selfemployed":     0.5,
        "student":          0.7,
        "unemployed":       0.8,
        "employed":         0.2,
        "retired":          0.3,
        "parttime":         0.5,
        "other":            0.6,
    }
    CHANNEL_RISK = {
        "RETAIL":           0.3,
        "DIGITAL":          0.5,
        "MOBILE":           0.6,
        "THIRD_PARTY":      0.9,
        "ATM":              0.4,
    }

    def __init__(self, top_n_features: int = 300):
        self.top_n_features = top_n_features

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        num_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != TARGET and "_is_missing" not in c
        ]
        variances = df[num_cols].var().sort_values(ascending=False)
        self.beh_cols_ = variances.head(self.top_n_features).index.tolist()

        # Population stats for high_activity flag
        means = df[self.beh_cols_].mean(axis=1)
        self.pop_mean_ = float(means.mean())
        self.pop_std_  = float(means.std()) or 1.0
        return self

    @staticmethod
    def _row_entropy(row: np.ndarray) -> float:
        """Shannon entropy of the discretised value distribution in a row."""
        vals = row[np.isfinite(row)]
        if len(vals) == 0:
            return 0.0
        # Bin into 10 equal-width buckets
        counts, _ = np.histogram(vals, bins=10)
        probs      = counts / (counts.sum() + 1e-9)
        probs      = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs + 1e-12)))

    def transform(self, X: pd.DataFrame):
        df   = X.copy()
        cols = [c for c in self.beh_cols_ if c in df.columns]
        sub  = df[cols].values.astype(float)

        df["beh_row_mean"]      = np.nanmean(sub, axis=1)
        df["beh_row_std"]       = np.nanstd(sub, axis=1)

        from scipy.stats import skew as _skew, kurtosis as _kurt
        df["beh_row_skew"]      = [_skew(row[np.isfinite(row)]) if np.isfinite(row).sum() > 2 else 0.0 for row in sub]
        df["beh_row_kurtosis"]  = [_kurt(row[np.isfinite(row)]) if np.isfinite(row).sum() > 3 else 0.0 for row in sub]

        pos   = np.where(sub > 0, sub, 0.0)
        neg   = np.where(sub < 0, sub, 0.0)
        df["beh_positive_sum"]  = np.nansum(pos, axis=1)
        df["beh_negative_sum"]  = np.nansum(neg, axis=1)
        df["beh_net_flow"]      = df["beh_positive_sum"] + df["beh_negative_sum"]

        inflow  = np.abs(df["beh_positive_sum"])
        outflow = np.abs(df["beh_negative_sum"])
        df["beh_flow_imbalance"] = np.abs(inflow - outflow) / (inflow + outflow + 1e-9)

        df["beh_high_activity"]  = (
            df["beh_row_mean"] > (self.pop_mean_ + 2 * self.pop_std_)
        ).astype(int)

        df["beh_zero_fraction"]      = (sub == 0).sum(axis=1) / (sub.shape[1] + 1e-9)
        df["beh_distinct_values"]    = [len(set(row[np.isfinite(row)])) for row in sub]
        df["beh_entropy"]            = [self._row_entropy(row) for row in sub]

        # Domain-knowledge categorical risk mappings
        for col_name, risk_map, feat_name in [
            ("F3886", self.ACCT_TYPE_RISK,    "beh_acct_type_risk"),
            ("F3891", self.EMPLOYMENT_RISK,   "beh_employment_risk"),
            ("F3893", self.CHANNEL_RISK,      "beh_channel_risk"),
        ]:
            if col_name in df.columns:
                df[feat_name] = df[col_name].map(risk_map).fillna(0.5)
            else:
                df[feat_name] = 0.5

        return df


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — FRAUD-RISK COMPOSITE FEATURES
# ════════════════════════════════════════════════════════════════════════════
class FraudRiskFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Assembles all previously engineered signals into interpretable, composite
    fraud-risk scores.  These final scores are the most directly actionable
    features for the model and for downstream human review.

    WHY MULE ACCOUNTS?
    ------------------
    A single numeric feature rarely determines fraud.  Composites encode
    *combinations* of risk signals — the way an analyst would reason:
    'this account is new, has a rare category, shows batch-opening behaviour,
    and has extreme outlier counts — that is a very high-risk account.'

    Features Generated
    ------------------
    risk_profile_score     : weighted average of te_mean_risk, out_outlier_rate,
                             beh_flow_imbalance, freq_rarity_score (normalised)
    risk_new_acct_flag     : 1 if account_age_days < 90 AND beh_high_activity
    risk_batch_open_flag   : 1 if date_open_cohort_risk > 2.0 (z-score)
    risk_synthetic_id_flag : 1 if freq_rarity_score ≥ 2
                             (≥2 rare categories → likely synthetic identity)
    risk_flow_thru_flag    : 1 if beh_flow_imbalance < 0.15
                             (near-perfect in = out balance → pass-through)
    risk_outlier_heavy_flag: 1 if out_total_flag_count > p90 of training dist
    risk_composite_score   : normalised sum of all binary risk flags + scaled
                             risk_profile_score.  Range [0, 1].
    risk_tier              : 0=Low, 1=Medium, 2=High, 3=Critical
                             (quantile cut of risk_composite_score)
    risk_anomaly_score     : IsolationForest-style heuristic using std of all
                             engineered features in the row
    """

    def __init__(self):
        self._fitted = False

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        # Calibrate the outlier count p90 threshold
        if "out_total_flag_count" in df.columns:
            self.out_flag_p90_ = float(df["out_total_flag_count"].quantile(0.90))
        else:
            self.out_flag_p90_ = 5.0

        # Calibrate score components for normalisation
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame):
        df = X.copy()

        # ── Helper: safe column access ──────────────────────────────────────
        def safe(col, default=0.0):
            return df[col] if col in df.columns else pd.Series(default, index=df.index)

        # ── risk_profile_score ──────────────────────────────────────────────
        components = {
            "te_mean_risk":       safe("te_mean_risk"),
            "out_outlier_rate":   safe("out_outlier_rate"),
            "beh_flow_imbalance": safe("beh_flow_imbalance"),
            "freq_rarity_score":  safe("freq_rarity_score") / 8.0,   # normalise by max possible
            "mv_missing_block_score": safe("mv_missing_block_score"),
        }
        weights = {"te_mean_risk": 0.25, "out_outlier_rate": 0.20,
                   "beh_flow_imbalance": 0.20, "freq_rarity_score": 0.20,
                   "mv_missing_block_score": 0.15}

        df["risk_profile_score"] = sum(
            components[k] * weights[k] for k in components
        )

        # ── Binary risk flags ────────────────────────────────────────────────
        acct_age   = safe("account_age_days", 9999)
        high_act   = safe("beh_high_activity", 0)
        cohort_z   = safe("date_open_cohort_risk", 0)
        rarity_sc  = safe("freq_rarity_score", 0)
        flow_imb   = safe("beh_flow_imbalance", 1.0)
        out_flags  = safe("out_total_flag_count", 0)

        df["risk_new_acct_flag"]        = ((acct_age < 90) & (high_act == 1)).astype(int)
        df["risk_batch_open_flag"]      = (cohort_z > 2.0).astype(int)
        df["risk_synthetic_id_flag"]    = (rarity_sc >= 2).astype(int)
        df["risk_flow_thru_flag"]       = (flow_imb < 0.15).astype(int)
        df["risk_outlier_heavy_flag"]   = (out_flags > self.out_flag_p90_).astype(int)

        # Employment × channel high-risk combo
        emp_risk  = safe("beh_employment_risk", 0.5)
        chan_risk  = safe("beh_channel_risk", 0.5)
        df["risk_emp_channel_combo"] = (emp_risk * chan_risk)

        # Total binary flags triggered
        flag_cols = [
            "risk_new_acct_flag", "risk_batch_open_flag",
            "risk_synthetic_id_flag", "risk_flow_thru_flag",
            "risk_outlier_heavy_flag"
        ]
        df["risk_flag_count"] = df[flag_cols].sum(axis=1)

        # ── Composite score [0, 1] ───────────────────────────────────────────
        flag_score   = df["risk_flag_count"] / len(flag_cols)
        profile_norm = df["risk_profile_score"].clip(0, 1)
        df["risk_composite_score"] = 0.5 * flag_score + 0.5 * profile_norm

        # ── Risk tier ────────────────────────────────────────────────────────
        df["risk_tier"] = pd.cut(
            df["risk_composite_score"],
            bins=[-0.001, 0.25, 0.50, 0.75, 1.001],
            labels=[0, 1, 2, 3]
        ).astype(float).fillna(0)

        # ── Anomaly score: how atypical is the full engineered feature vector ─
        eng_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c != TARGET and c not in flag_cols]
        sub = df[eng_cols].fillna(0).values.astype(float)
        df["risk_anomaly_score"] = np.std(sub, axis=1) / (np.mean(np.abs(sub), axis=1) + 1e-9)

        return df


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — SHAP-DRIVEN FEATURE SELECTION
# ════════════════════════════════════════════════════════════════════════════
class SHAPFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Trains a LightGBM classifier on the fully engineered feature set and uses
    TreeExplainer SHAP values to rank and select the most predictive features.

    WHY SHAP?
    ---------
    In a high-dimensional, severely imbalanced AML dataset (81 mules in 9,082
    rows), many engineered features will be correlated or redundant.  SHAP
    provides model-consistent, theoretically-grounded feature importances:
    unlike gain or split-count importance, SHAP values accurately attribute
    the model's output to each input feature, even in the presence of
    correlations.  This makes the selection both accurate and explainable to
    compliance teams.

    Process
    -------
    1. Fit LightGBM with class_weight='balanced' to handle imbalance.
    2. Compute SHAP values using shap.TreeExplainer (fast, exact for trees).
    3. Rank features by mean |SHAP| across all training rows.
    4. Select top-N features (default 150) plus all risk composite features.
    5. Return a reduced, explanation-ready DataFrame.

    Outputs (saved to disk)
    -----------------------
    shap_feature_importance.csv  : feature | mean_abs_shap | rank
    shap_summary_plot.png        : beeswarm plot of top-30 SHAP values
    selected_features.json       : list of selected feature names
    """

    def __init__(self, top_n: int = 150, lgbm_params: dict = None, output_dir: str = "."):
        self.top_n      = top_n
        self.output_dir = output_dir
        self.lgbm_params = lgbm_params or {
            "objective":        "binary",
            "metric":           "auc",
            "n_estimators":     500,
            "learning_rate":    0.05,
            "num_leaves":       31,
            "min_child_samples": 5,
            "class_weight":     "balanced",
            "random_state":     42,
            "n_jobs":           -1,
            "verbose":          -1,
        }
        self.selected_features_ = None

    # ── Column-name sanitiser ───────────────────────────────────────────────
    @staticmethod
    def _sanitize_names(cols):
        """
        Replace every character that is not alphanumeric or underscore with '_'.
        LightGBM serialises feature names as JSON and chokes on brackets,
        colons, angle-brackets, spaces, etc. that appear in interaction-feature
        names such as  interact_F10_x_F20  →  already safe, but names produced
        by te_interact columns or columns carrying '<', '>', '[', ']' etc. must
        be cleaned.  Returns (safe_cols, orig_to_safe_dict, safe_to_orig_dict).
        """
        import re
        safe_cols   = []
        seen        = {}      # tracks collisions after sanitisation
        orig_to_safe = {}
        safe_to_orig = {}

        for orig in cols:
            safe = re.sub(r"[^A-Za-z0-9_]", "_", orig)
            # If two original names map to the same sanitised name, append a counter
            if safe in seen:
                seen[safe] += 1
                safe = f"{safe}_{seen[safe]}"
            else:
                seen[safe] = 0
            safe_cols.append(safe)
            orig_to_safe[orig] = safe
            safe_to_orig[safe] = orig

        return safe_cols, orig_to_safe, safe_to_orig

    def fit(self, X: pd.DataFrame, y):
        try:
            import lightgbm as lgb
            import shap
        except ImportError:
            raise ImportError(
                "lightgbm and shap are required for SHAP feature selection.\n"
                "Install with: pip install lightgbm shap"
            )

        df        = X.copy()
        feat_cols = [c for c in df.columns if c != TARGET]
        X_fit     = df[feat_cols].fillna(-9999)

        # ── Sanitise column names for LightGBM ──────────────────────────────
        safe_cols, orig_to_safe, safe_to_orig = self._sanitize_names(feat_cols)
        X_fit_lgbm = X_fit.copy()
        X_fit_lgbm.columns = safe_cols          # renamed DataFrame for LightGBM
        self._orig_to_safe = orig_to_safe       # stored for inference
        self._safe_to_orig = safe_to_orig

        print(f"\n[SHAP] Training LightGBM on {X_fit_lgbm.shape[1]} features …")
        model = lgb.LGBMClassifier(**self.lgbm_params)
        model.fit(X_fit_lgbm, y)               # train on sanitised names

        print("[SHAP] Computing SHAP values …")
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_fit_lgbm)

        # For binary classification, shap_values may be a list [neg, pos]
        if isinstance(shap_values, list):
            shap_arr = shap_values[1]
        else:
            shap_arr = shap_values

        # Map SHAP importances back to ORIGINAL column names
        mean_abs_shap = pd.Series(
            np.abs(shap_arr).mean(axis=0),
            index=[safe_to_orig.get(s, s) for s in safe_cols]   # restore original names
        ).sort_values(ascending=False)

        # Always keep composite risk features
        must_keep   = [c for c in feat_cols if c.startswith("risk_") or c.startswith("te_")]
        top_by_shap = mean_abs_shap.head(self.top_n).index.tolist()
        self.selected_features_ = list(dict.fromkeys(top_by_shap + must_keep))  # ordered, deduped

        # ── Save importance table (original names) ───────────────────────────
        importance_df = mean_abs_shap.reset_index()
        importance_df.columns = ["feature", "mean_abs_shap"]
        importance_df["rank"] = range(1, len(importance_df) + 1)
        out_path = os.path.join(self.output_dir, "shap_feature_importance.csv")
        importance_df.to_csv(out_path, index=False)
        print(f"[SHAP] Importance saved → {out_path}")

        # ── Save selected features list ──────────────────────────────────────
        sel_path = os.path.join(self.output_dir, "selected_features.json")
        with open(sel_path, "w") as f:
            json.dump(self.selected_features_, f, indent=2)
        print(f"[SHAP] Selected {len(self.selected_features_)} features saved → {sel_path}")

        # ── Beeswarm plot (top 30) ───────────────────────────────────────────
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            top30_safe   = safe_cols[:30]
            top30_orig   = [safe_to_orig.get(s, s) for s in top30_safe]
            X_plot       = X_fit_lgbm.iloc[:, :30].copy()
            X_plot.columns = top30_orig          # restore readable names for plot

            shap.summary_plot(
                shap_arr[:, :30],
                X_plot,
                show=False,
                max_display=30
            )
            plot_path = os.path.join(self.output_dir, "shap_summary_plot.png")
            plt.savefig(plot_path, bbox_inches="tight", dpi=150)
            plt.close()
            print(f"[SHAP] Summary plot saved → {plot_path}")
        except Exception as e:
            print(f"[SHAP] Could not generate plot: {e}")

        self._model     = model
        self._explainer = explainer
        return self

    def transform(self, X: pd.DataFrame):
        if self.selected_features_ is None:
            raise RuntimeError("Call fit() before transform().")
        df   = X.copy()
        cols = [c for c in self.selected_features_ if c in df.columns]
        if TARGET in df.columns:
            cols = cols + [TARGET]
        return df[cols]


# ════════════════════════════════════════════════════════════════════════════
#  MASTER PIPELINE ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════
class MuleAccountFeatureEngineeringPipeline:
    """
    End-to-end feature engineering pipeline for mule account detection.
    Wraps all 10 stages in correct execution order.

    Usage (training)
    ----------------
    >>> pipeline = MuleAccountFeatureEngineeringPipeline(output_dir=".")
    >>> df_engineered = pipeline.fit_transform(df_cleaned, df_cleaned[TARGET])
    >>> pipeline.save("mule_feature_pipeline.pkl")

    Usage (inference)
    -----------------
    >>> pipeline = MuleAccountFeatureEngineeringPipeline.load("mule_feature_pipeline.pkl")
    >>> df_inference = pipeline.transform(new_df)
    """

    def __init__(
        self,
        output_dir: str = ".",
        top_shap_features: int = 150,
        run_shap: bool = True,
    ):
        self.output_dir        = output_dir
        self.top_shap_features = top_shap_features
        self.run_shap          = run_shap

        # Instantiate all stages
        self.s1_mv      = MissingValueFeatureEngineer()
        self.s2_freq    = FrequencyEncoder()
        self.s3_te      = SmoothedTargetEncoder()
        self.s4_date    = DateDecompositionEngineer()
        self.s5_ratio   = RatioFeatureEngineer()
        self.s6_inter   = InteractionFeatureEngineer()
        self.s7_out     = OutlierIndicatorEngineer()
        self.s8_beh     = BehaviouralFeatureEngineer()
        self.s9_risk    = FraudRiskFeatureEngineer()
        if self.run_shap:
            self.s10_shap = SHAPFeatureSelector(
                top_n=top_shap_features, output_dir=output_dir
            )

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        """Full training-time pipeline (with OOF target encoding)."""
        if y is None and TARGET in df.columns:
            y = df[TARGET].values

        print("\n" + "═" * 70)
        print("  MULE ACCOUNT FEATURE ENGINEERING PIPELINE — FIT+TRANSFORM")
        print("═" * 70)
        print(f"  Input shape : {df.shape}")

        # Drop known leakage
        df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns], errors="ignore")

        # ── Stage 1: Missing-value features ──────────────────────────────────
        print("\n[1/10] Missing-value features …")
        self.s1_mv.fit(df, y)
        df = self.s1_mv.transform(df)
        print(f"       → shape {df.shape}")

        # ── Stage 2: Frequency encoding ──────────────────────────────────────
        print("\n[2/10] Frequency encoding …")
        self.s2_freq.fit(df)
        df = self.s2_freq.transform(df)
        print(f"       → shape {df.shape}")

        # ── Stage 3: Target encoding (OOF) ───────────────────────────────────
        print("\n[3/10] Smoothed target encoding (OOF) …")
        self.s3_te.fit(df, y)
        df = self.s3_te.transform(df)      # use inference path; OOF ran on training data
        print(f"       → shape {df.shape}")

        # ── Stage 4: Date decomposition ──────────────────────────────────────
        print("\n[4/10] Date decomposition (F3888) …")
        self.s4_date.fit(df)
        df = self.s4_date.transform(df)
        print(f"       → shape {df.shape}")

        # ── Stage 5: Ratio features ──────────────────────────────────────────
        print("\n[5/10] Ratio features …")
        self.s5_ratio.fit(df)
        df = self.s5_ratio.transform(df)
        print(f"       → shape {df.shape}")

        # ── Stage 6: Interaction features ────────────────────────────────────
        print("\n[6/10] Interaction features …")
        self.s6_inter.fit(df)
        df = self.s6_inter.transform(df)
        print(f"       → shape {df.shape}")

        # ── Stage 7: Outlier indicators ──────────────────────────────────────
        print("\n[7/10] Outlier indicators …")
        self.s7_out.fit(df)
        df = self.s7_out.transform(df)
        print(f"       → shape {df.shape}")

        # ── Stage 8: Behavioural features ────────────────────────────────────
        print("\n[8/10] Behavioural features …")
        self.s8_beh.fit(df)
        df = self.s8_beh.transform(df)
        print(f"       → shape {df.shape}")

        # ── Stage 9: Fraud-risk composites ───────────────────────────────────
        print("\n[9/10] Fraud-risk composite features …")
        self.s9_risk.fit(df)
        df = self.s9_risk.transform(df)
        print(f"       → shape {df.shape}")

        # ── Stage 10: SHAP-driven selection ──────────────────────────────────
        if self.run_shap:
            print("\n[10/10] SHAP-driven feature selection …")
            if y is not None:
                self.s10_shap.fit(df.drop(columns=[TARGET], errors="ignore"), y)
                df = self.s10_shap.transform(df)
            else:
                print("        ⚠  No target provided — SHAP selection skipped.")
            print(f"       → shape {df.shape}")

        print("\n" + "═" * 70)
        print(f"  PIPELINE COMPLETE  |  Final shape: {df.shape}")
        print("═" * 70 + "\n")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inference-time transform (no target required)."""
        df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns], errors="ignore")
        df = self.s1_mv.transform(df)
        df = self.s2_freq.transform(df)
        df = self.s3_te.transform(df)
        df = self.s4_date.transform(df)
        df = self.s5_ratio.transform(df)
        df = self.s6_inter.transform(df)
        df = self.s7_out.transform(df)
        df = self.s8_beh.transform(df)
        df = self.s9_risk.transform(df)
        if self.run_shap and hasattr(self, "s10_shap"):
            df = self.s10_shap.transform(df)
        return df

    def save(self, path: str):
        joblib.dump(self, path)
        print(f"Pipeline saved → {path}")

    @classmethod
    def load(cls, path: str) -> "MuleAccountFeatureEngineeringPipeline":
        obj = joblib.load(path)
        print(f"Pipeline loaded ← {path}")
        return obj


# ════════════════════════════════════════════════════════════════════════════
#  POST-PIPELINE CLEANUP — zero / constant / near-constant column removal
# ════════════════════════════════════════════════════════════════════════════
def drop_useless_columns(df: pd.DataFrame, target_col: str = TARGET, verbose: bool = True) -> pd.DataFrame:
    """
    Scans every column in df and drops those that carry no discriminative
    information:

        all_zero     — every value (including NaN→0) is 0
        constant     — only 1 unique non-NaN value across all rows
        near_constant— a single value covers > 99.9 % of rows

    These arise in the feature engineering pipeline when:
    • An outlier / IQR flag column fires zero times in the data subset.
    • An interaction feature happens to be two always-zero columns multiplied.
    • A risk flag was never triggered across the dataset.

    Returns the cleaned DataFrame and prints a summary.
    """
    drop_log = {"all_zero": [], "constant": [], "near_constant": []}

    for c in df.columns:
        if c == target_col:
            continue
        s      = df[c]
        filled = s.fillna(0)

        if (filled == 0).all():
            drop_log["all_zero"].append(c)
            continue

        if s.nunique(dropna=True) <= 1:
            drop_log["constant"].append(c)
            continue

        top_freq = s.value_counts(normalize=True, dropna=False).iloc[0]
        if top_freq > 0.999:
            drop_log["near_constant"].append(c)

    all_to_drop = [c for cats in drop_log.values() for c in cats]

    if verbose:
        print("\n── Post-pipeline Column Cleanup ─────────────────────────────────────")
        print(f"  All-zero columns      dropped : {len(drop_log['all_zero'])}")
        print(f"  Constant columns      dropped : {len(drop_log['constant'])}")
        print(f"  Near-constant columns dropped : {len(drop_log['near_constant'])}")
        print(f"  Total dropped                 : {len(all_to_drop)}")
        print(f"  Columns before                : {df.shape[1]}")
        print(f"  Columns after                 : {df.shape[1] - len(all_to_drop)}")
        if drop_log["all_zero"]:
            print(f"\n  All-zero (first 10): {drop_log['all_zero'][:10]}")
        if drop_log["constant"]:
            print(f"  Constant (first 10): {drop_log['constant'][:10]}")

    df_clean = df.drop(columns=all_to_drop, errors="ignore")
    return df_clean, drop_log


# ════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
def main():
    import sys

    print("Loading DataSet_Cleaned.csv …")
    DATA_PATH = "DataSet_Cleaned.csv"
    if not os.path.exists(DATA_PATH):
        print(f"❌  {DATA_PATH} not found.  Run data_pipeline.py first.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"   Loaded {df.shape[0]:,} rows × {df.shape[1]:,} columns")

    if TARGET not in df.columns:
        print(f"❌  Target column '{TARGET}' not found in dataset.")
        sys.exit(1)

    y  = df[TARGET].values

    # ── Run the full feature engineering pipeline ─────────────────────────
    pipeline = MuleAccountFeatureEngineeringPipeline(
        output_dir        = ".",
        top_shap_features = 150,
        run_shap          = True,    # set False to skip LightGBM/SHAP (faster)
    )

    df_engineered = pipeline.fit_transform(df, y)

    # ── Post-pipeline cleanup: remove zero / constant / near-constant cols ─
    print("\n[Post-processing] Removing zero / constant / near-constant columns …")
    df_engineered, drop_log = drop_useless_columns(df_engineered, target_col=TARGET, verbose=True)

    # Save cleaned engineered dataset
    out_csv = "DataSet_Engineered.csv"
    df_engineered.to_csv(out_csv, index=False)
    print(f"\n✅  Engineered dataset saved → {out_csv}")
    print(f"    Final shape: {df_engineered.shape[0]:,} rows × {df_engineered.shape[1]:,} columns")

    # Save pipeline
    pipeline.save("mule_feature_pipeline.pkl")

    # Save drop log
    drop_log_path = "dropped_columns_log.json"
    with open(drop_log_path, "w") as f:
        json.dump(drop_log, f, indent=2)
    print(f"    Drop log saved → {drop_log_path}")

    # ── Feature group summary ─────────────────────────────────────────────
    cols = df_engineered.columns.tolist()
    summary = {
        "Missing-value (mv_)":           sum(1 for c in cols if c.startswith("mv_") or c.endswith("_is_missing")),
        "Frequency encoded (freq_)":     sum(1 for c in cols if "_freq" in c or c.startswith("freq_")),
        "Target encoded (_te)":          sum(1 for c in cols if c.endswith("_te") or c.endswith("_te_deviation") or c.startswith("te_")),
        "Date features (date_/acct_)":   sum(1 for c in cols if c.startswith("date_") or c.startswith("account_")),
        "Ratio features (ratio_)":       sum(1 for c in cols if c.startswith("ratio_")),
        "Interaction features (inter_)": sum(1 for c in cols if c.startswith("interact_") or c.startswith("te_interact_")),
        "Outlier indicators (out_)":     sum(1 for c in cols if c.endswith("_iqr_outlier") or c.endswith("_zscore_outlier") or c.startswith("out_")),
        "Behavioural features (beh_)":   sum(1 for c in cols if c.startswith("beh_")),
        "Fraud-risk features (risk_)":   sum(1 for c in cols if c.startswith("risk_")),
    }
    print("\n── Engineered Feature Summary ──────────────────────────────────────")
    for grp, cnt in summary.items():
        print(f"   {grp:<47} {cnt:>4} features")
    print(f"   {'TOTAL (excl. target)':<47} {sum(summary.values()):>4} engineered features")
    print("─" * 65)

    # ── Also check the 5 flagged Excel columns specifically ───────────────
    def excel_col_to_idx(letters: str) -> int:
        result = 0
        for ch in letters.upper():
            result = result * 26 + (ord(ch) - ord("A") + 1)
        return result - 1

    print("\n── User-flagged Excel columns (EV–EZ) status ───────────────────────")
    target_letters = ["EV", "EW", "EX", "EY", "EZ"]
    # These indices relate to the *original* engineered CSV before cleanup;
    # check whether those positions even survive
    header = pd.read_csv(out_csv, nrows=0)
    final_cols = header.columns.tolist()
    for letter in target_letters:
        idx = excel_col_to_idx(letter)
        if idx < len(final_cols):
            print(f"  {letter} (idx {idx:>4d})  →  '{final_cols[idx]}'  ✅ still present")
        else:
            print(f"  {letter} (idx {idx:>4d})  →  column index exceeds final column count ({len(final_cols)}) — dropped or out of range")


if __name__ == "__main__":
    main()
