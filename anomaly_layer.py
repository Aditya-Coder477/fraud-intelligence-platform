"""
================================================================================
  ANOMALY DETECTION LAYER — BANKING FRAUD / MULE ACCOUNT DETECTION
  Detectors : Isolation Forest · Local Outlier Factor · Autoencoder
  Score     : [0, 100]  (100 = maximally anomalous)
  Fusion    : Anomaly Score + Classification Score → Final Risk Score
================================================================================

TRAINING PHILOSOPHY
───────────────────
All three detectors are trained on LEGITIMATE accounts ONLY (y=0).
This models the "normal" distribution. Any account that deviates from it
— including novel mule typologies never seen during training — scores high.

This makes anomaly detection COMPLEMENTARY to the classifier:
  · Classifier  → catches known mule patterns (labelled learning)
  · Anomaly layer → catches unknown / new mule patterns (unsupervised)

SCORE SEMANTICS
───────────────
  0   → Perfectly normal (identical to average legitimate account)
  50  → Moderately unusual
  80+ → Highly anomalous — warrants investigation
  100 → Most extreme outlier in training distribution

FUSION FORMULA
──────────────
  fusion_score = α × classif_score + (1 - α) × anomaly_score
  Where  classif_score = mule_prob × 100,  anomaly_score ∈ [0, 100]
  Default α = 0.65  (classifier-weighted; tune for your false-positive tolerance)

RISK TIERS
──────────
  CRITICAL  ≥ 80   → Immediate SAR / block
  HIGH      60–79  → Priority review queue
  MEDIUM    40–59  → Enhanced monitoring
  LOW       < 40   → Standard processing
"""

# ── stdlib ───────────────────────────────────────────────────────────────────
import os
import sys
import json
import warnings
import logging
import re
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("AnomalyLayer")

# ── third-party ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
)

# PyTorch (optional — falls back to sklearn MLPRegressor)
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    log.info("PyTorch detected — using neural Autoencoder")
except ImportError:
    from sklearn.neural_network import MLPRegressor
    log.warning("PyTorch not found — falling back to sklearn MLPRegressor Autoencoder. "
                "Install with: pip install torch")

# ════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
TARGET       = "F3924"
RANDOM_STATE = 42

BASE_DIR     = Path(".")
AD_DIR       = BASE_DIR / "anomaly_detection"
AD_MODEL_DIR = AD_DIR / "models"
AD_PLOT_DIR  = AD_DIR / "plots"
AD_MODEL_DIR.mkdir(parents=True, exist_ok=True)
AD_PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Detector weights for ensemble anomaly score
W_ISO  = 0.30
W_LOF  = 0.25
W_AE   = 0.45

# Fusion weight (α): how much to weight the classifier vs anomaly score
FUSION_ALPHA = 0.65


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 0 — UTILITIES
# ════════════════════════════════════════════════════════════════════════════
def sanitize_col_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [re.sub(r"[^A-Za-z0-9_]", "_", c) for c in df.columns]
    seen, new_cols = {}, []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    df.columns = new_cols
    return df


def percentile_normalize(
    raw_scores: np.ndarray,
    p_low: float,
    p_high: float,
    invert: bool = False,
) -> np.ndarray:
    """
    Map raw scores to [0, 100] using training-set percentile bounds.
    If invert=True, lower raw score = higher anomaly (used for IF / LOF).
    """
    if invert:
        raw_scores = -raw_scores
        p_low, p_high = -p_high, -p_low

    norm = np.clip((raw_scores - p_low) / (p_high - p_low + 1e-9), 0.0, 1.0)
    return norm * 100.0


def risk_tier(score: float) -> str:
    if score >= 80: return "CRITICAL"
    if score >= 60: return "HIGH"
    if score >= 40: return "MEDIUM"
    return "LOW"


def load_data() -> tuple:
    """Load engineered dataset; return X_legit, X_all, y."""
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
        log.error("No dataset found.")
        sys.exit(1)

    df.drop(columns=[c for c in ["Unnamed: 0", "index"] if c in df.columns], inplace=True)
    df = sanitize_col_names(df)

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])
    X = X.fillna(X.median(numeric_only=True)).fillna(0).astype(np.float32)

    log.info(f"  Legit : {(y==0).sum():,}  |  Mule: {(y==1).sum():,}")
    return X[y == 0], X, y


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — ISOLATION FOREST
# ════════════════════════════════════════════════════════════════════════════
class IsolationForestDetector:
    """
    Isolation Forest trained on legitimate accounts only.

    score_samples() returns negative scores where more negative = more anomalous.
    We invert and normalise to [0, 100].
    """

    def __init__(
        self,
        n_estimators: int = 300,
        contamination: float = 0.01,
        max_samples: str = "auto",
    ):
        self.model = IsolationForest(
            n_estimators   = n_estimators,
            contamination  = contamination,
            max_samples    = max_samples,
            random_state   = RANDOM_STATE,
            n_jobs         = -1,
        )
        self._p_low  = None
        self._p_high = None

    def fit(self, X_legit: np.ndarray) -> "IsolationForestDetector":
        log.info(f"  [IF] Training on {len(X_legit):,} legitimate accounts …")
        self.model.fit(X_legit)

        # Compute normalisation bounds on training data
        raw = self.model.score_samples(X_legit)   # negative: anomalous
        self._p_low  = float(np.percentile(raw, 1))    # most anomalous in legit
        self._p_high = float(np.percentile(raw, 99))   # most normal in legit
        log.info(f"  [IF] Done. Score range [{self._p_low:.4f}, {self._p_high:.4f}]")
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores in [0, 100]. 100 = maximally anomalous."""
        raw = self.model.score_samples(X)
        return percentile_normalize(raw, self._p_low, self._p_high, invert=True)

    def predict_labels(self, X: np.ndarray, threshold: float = 50.0) -> np.ndarray:
        """Return 1 (anomaly) or 0 (normal) at given score threshold."""
        return (self.score(X) >= threshold).astype(int)


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — LOCAL OUTLIER FACTOR
# ════════════════════════════════════════════════════════════════════════════
class LOFDetector:
    """
    Local Outlier Factor with novelty=True so it can score unseen data.

    Trained on legitimate accounts only.
    LOF score < 1 = inlier, >> 1 = outlier.
    """

    def __init__(self, n_neighbors: int = 20, contamination: float = 0.01):
        self.model = LocalOutlierFactor(
            n_neighbors   = n_neighbors,
            contamination = contamination,
            novelty       = True,         # REQUIRED for scoring new data
            n_jobs        = -1,
            algorithm     = "auto",
            metric        = "euclidean",
        )
        self._scaler = RobustScaler()
        self._p_low  = None
        self._p_high = None

    def fit(self, X_legit: np.ndarray) -> "LOFDetector":
        log.info(f"  [LOF] Training on {len(X_legit):,} legitimate accounts …")
        X_scaled = self._scaler.fit_transform(X_legit)
        self.model.fit(X_scaled)

        # Calibrate normalisation bounds
        raw = self.model.score_samples(X_scaled)   # negative: anomalous
        self._p_low  = float(np.percentile(raw, 1))
        self._p_high = float(np.percentile(raw, 99))
        log.info(f"  [LOF] Done. Score range [{self._p_low:.4f}, {self._p_high:.4f}]")
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores in [0, 100]."""
        X_scaled = self._scaler.transform(X)
        raw      = self.model.score_samples(X_scaled)
        return percentile_normalize(raw, self._p_low, self._p_high, invert=True)

    def predict_labels(self, X: np.ndarray, threshold: float = 50.0) -> np.ndarray:
        return (self.score(X) >= threshold).astype(int)


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 3A — PYTORCH AUTOENCODER
# ════════════════════════════════════════════════════════════════════════════
if TORCH_AVAILABLE:
    class _AutoencoderNet(nn.Module):
        """
        Symmetric Autoencoder with SELU activations.
        Architecture: n_in → 128 → 64 → 32 → 16 → 32 → 64 → 128 → n_in
        """

        def __init__(self, n_in: int, dropout: float = 0.2):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_in,  128), nn.SELU(), nn.Dropout(dropout),
                nn.Linear(128,    64), nn.SELU(), nn.Dropout(dropout),
                nn.Linear(64,     32), nn.SELU(),
                nn.Linear(32,     16), nn.SELU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(16,     32), nn.SELU(),
                nn.Linear(32,     64), nn.SELU(), nn.Dropout(dropout),
                nn.Linear(64,    128), nn.SELU(), nn.Dropout(dropout),
                nn.Linear(128, n_in),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

        def encode(self, x):
            return self.encoder(x)


class AutoencoderDetector:
    """
    Autoencoder-based anomaly detector.

    Trains to reconstruct legitimate account feature vectors.
    High reconstruction error = unusual = anomalous.

    Backend: PyTorch (preferred) or sklearn MLPRegressor (fallback).
    """

    def __init__(
        self,
        n_epochs: int   = 100,
        batch_size: int = 64,
        lr: float       = 1e-3,
        patience: int   = 10,
        val_frac: float = 0.15,
    ):
        self.n_epochs   = n_epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.patience   = patience
        self.val_frac   = val_frac
        self._scaler    = StandardScaler()
        self._net       = None
        self._mlp       = None
        self._p_low     = None
        self._p_high    = None
        self._n_in      = None
        self._use_torch = TORCH_AVAILABLE

    # ── Torch training ────────────────────────────────────────────────────
    def _train_torch(self, X_scaled: np.ndarray):
        n   = len(X_scaled)
        n_v = int(n * self.val_frac)
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.permutation(n)
        X_tr = torch.tensor(X_scaled[idx[n_v:]], dtype=torch.float32)
        X_v  = torch.tensor(X_scaled[idx[:n_v]], dtype=torch.float32)

        self._net = _AutoencoderNet(self._n_in)
        opt       = optim.Adam(self._net.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        crit      = nn.MSELoss()
        loader    = DataLoader(TensorDataset(X_tr), batch_size=self.batch_size, shuffle=True)

        best_val, patience_count, best_state = float("inf"), 0, None
        history = []

        for epoch in range(1, self.n_epochs + 1):
            self._net.train()
            train_loss = 0.0
            for (batch,) in loader:
                opt.zero_grad()
                loss = crit(self._net(batch), batch)
                loss.backward()
                opt.step()
                train_loss += loss.item() * len(batch)
            train_loss /= len(X_tr)

            self._net.eval()
            with torch.no_grad():
                val_loss = crit(self._net(X_v), X_v).item()

            scheduler.step(val_loss)
            history.append({"epoch": epoch, "train": train_loss, "val": val_loss})

            if val_loss < best_val:
                best_val     = val_loss
                best_state   = {k: v.clone() for k, v in self._net.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1

            if epoch % 20 == 0:
                log.info(f"    [AE] Epoch {epoch:>3d}/{self.n_epochs}  "
                         f"train={train_loss:.6f}  val={val_loss:.6f}")

            if patience_count >= self.patience:
                log.info(f"    [AE] Early stopping at epoch {epoch}")
                break

        # Restore best weights
        self._net.load_state_dict(best_state)
        self._net.eval()
        log.info(f"    [AE] Best val MSE: {best_val:.6f}")

        # Plot training curve
        self._plot_training_curve(history)

    def _plot_training_curve(self, history: list):
        epochs     = [h["epoch"] for h in history]
        train_loss = [h["train"] for h in history]
        val_loss   = [h["val"]   for h in history]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(epochs, train_loss, label="Train MSE", color="#3498db", lw=2)
        ax.plot(epochs, val_loss,   label="Val MSE",   color="#e74c3c", lw=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Reconstruction MSE", fontsize=12)
        ax.set_title("Autoencoder Training Curve", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(AD_PLOT_DIR / "ae_training_curve.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ── sklearn fallback ──────────────────────────────────────────────────
    def _train_sklearn(self, X_scaled: np.ndarray):
        log.info("    [AE] Using sklearn MLPRegressor (PyTorch not available)")
        self._mlp = MLPRegressor(
            hidden_layer_sizes = (128, 64, 32, 16, 32, 64, 128),
            activation         = "relu",
            max_iter           = self.n_epochs,
            learning_rate_init = self.lr,
            random_state       = RANDOM_STATE,
            early_stopping     = True,
            validation_fraction= self.val_frac,
            n_iter_no_change   = self.patience,
            verbose            = False,
        )
        self._mlp.fit(X_scaled, X_scaled)  # autoencoder: input = target

    # ── Reconstruction error ──────────────────────────────────────────────
    def _reconstruction_error(self, X_scaled: np.ndarray) -> np.ndarray:
        """Per-sample mean squared reconstruction error."""
        if self._use_torch and self._net is not None:
            self._net.eval()
            X_t  = torch.tensor(X_scaled, dtype=torch.float32)
            with torch.no_grad():
                recon = self._net(X_t).numpy()
        else:
            recon = self._mlp.predict(X_scaled)
        return np.mean((X_scaled - recon) ** 2, axis=1)

    # ── Public API ────────────────────────────────────────────────────────
    def fit(self, X_legit: np.ndarray) -> "AutoencoderDetector":
        log.info(f"  [AE] Training on {len(X_legit):,} legitimate accounts …")
        self._n_in    = X_legit.shape[1]
        X_scaled      = self._scaler.fit_transform(X_legit)

        if self._use_torch:
            self._train_torch(X_scaled)
        else:
            self._train_sklearn(X_scaled)

        # Compute normalisation bounds
        errors       = self._reconstruction_error(X_scaled)
        self._p_low  = float(np.percentile(errors, 1))
        self._p_high = float(np.percentile(errors, 99))
        log.info(f"  [AE] Error range [{self._p_low:.6f}, {self._p_high:.6f}]")
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores in [0, 100]."""
        X_scaled = self._scaler.transform(X)
        errors   = self._reconstruction_error(X_scaled)
        return percentile_normalize(errors, self._p_low, self._p_high, invert=False)

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Return raw per-sample reconstruction MSE (unnormalized)."""
        return self._reconstruction_error(self._scaler.transform(X))

    def predict_labels(self, X: np.ndarray, threshold: float = 50.0) -> np.ndarray:
        return (self.score(X) >= threshold).astype(int)

    def save_weights(self, path: str):
        if self._use_torch and self._net is not None:
            torch.save(self._net.state_dict(), path)
            log.info(f"  [AE] Weights saved → {path}")

    def load_weights(self, path: str):
        if self._use_torch:
            self._net = _AutoencoderNet(self._n_in)
            self._net.load_state_dict(torch.load(path, map_location="cpu"))
            self._net.eval()


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — ENSEMBLE ANOMALY PIPELINE
# ════════════════════════════════════════════════════════════════════════════
class AnomalyDetectionPipeline:
    """
    Combines Isolation Forest, LOF, and Autoencoder into a single
    anomaly detection layer that produces a unified [0, 100] score.

    Usage
    ─────
    >>> pipeline = AnomalyDetectionPipeline()
    >>> pipeline.fit(X_legit)
    >>> scores = pipeline.score(X_new)                # [0, 100] per row
    >>> report = pipeline.predict(X_new)              # full DataFrame
    >>> fused  = pipeline.fuse(X_new, mule_prob)      # final risk score

    Persistence
    ───────────
    >>> pipeline.save()
    >>> pipeline = AnomalyDetectionPipeline.load()
    """

    def __init__(
        self,
        w_iso: float  = W_ISO,
        w_lof: float  = W_LOF,
        w_ae: float   = W_AE,
        fusion_alpha: float = FUSION_ALPHA,
        if_params: dict   = None,
        lof_params: dict  = None,
        ae_params: dict   = None,
    ):
        self.w_iso        = w_iso
        self.w_lof        = w_lof
        self.w_ae         = w_ae
        self.fusion_alpha = fusion_alpha
        self.feature_names = None

        if_params  = if_params  or {}
        lof_params = lof_params or {}
        ae_params  = ae_params  or {}

        self.iso = IsolationForestDetector(**if_params)
        self.lof = LOFDetector(**lof_params)
        self.ae  = AutoencoderDetector(**ae_params)

    # ── Fitting ───────────────────────────────────────────────────────────
    def fit(self, X_legit: pd.DataFrame) -> "AnomalyDetectionPipeline":
        """Train all detectors on legitimate accounts only."""
        log.info("\n" + "═"*60)
        log.info("  ANOMALY DETECTION — Training on legitimate accounts")
        log.info("═"*60)
        self.feature_names = X_legit.columns.tolist()
        X_arr = X_legit.values.astype(np.float32)

        self.iso.fit(X_arr)
        self.lof.fit(X_arr)
        self.ae.fit(X_arr)

        log.info("\n  ✅  All detectors trained.")
        return self

    # ── Scoring ───────────────────────────────────────────────────────────
    def _align(self, X: pd.DataFrame) -> np.ndarray:
        """Align columns to training feature list."""
        X = sanitize_col_names(X.copy())
        for c in self.feature_names:
            if c not in X.columns:
                X[c] = 0.0
        return X[self.feature_names].fillna(0).astype(np.float32).values

    def score_components(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return individual detector scores + ensemble score for each row.

        Returns DataFrame with columns:
          iso_score, lof_score, ae_score, anomaly_score
        """
        X_arr = self._align(X)
        iso_s = self.iso.score(X_arr)
        lof_s = self.lof.score(X_arr)
        ae_s  = self.ae.score(X_arr)
        ens_s = self.w_iso * iso_s + self.w_lof * lof_s + self.w_ae * ae_s

        return pd.DataFrame({
            "iso_score":     iso_s,
            "lof_score":     lof_s,
            "ae_score":      ae_s,
            "anomaly_score": np.clip(ens_s, 0, 100),
        })

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Return ensemble anomaly score [0, 100] for each row."""
        return self.score_components(X)["anomaly_score"].values

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return full anomaly prediction report."""
        scores = self.score_components(X)
        scores["anomaly_label"] = (scores["anomaly_score"] >= 50).astype(int)
        scores["anomaly_tier"]  = scores["anomaly_score"].apply(risk_tier)
        return scores

    def fuse(
        self,
        X: pd.DataFrame,
        mule_prob: np.ndarray,
        alpha: float = None,
    ) -> pd.DataFrame:
        """
        Fuse anomaly score with classifier probability.

        Parameters
        ──────────
        X         : feature DataFrame
        mule_prob : array of mule probabilities from the classifier [0, 1]
        alpha     : weight for classifier (1-alpha for anomaly). Default: self.fusion_alpha

        Returns DataFrame with:
          classif_score, anomaly_score, fusion_score, risk_tier,
          iso_score, lof_score, ae_score
        """
        alpha         = alpha if alpha is not None else self.fusion_alpha
        comp          = self.score_components(X)
        classif_score = np.clip(np.asarray(mule_prob) * 100, 0, 100)
        anomaly_score = comp["anomaly_score"].values
        fusion_score  = alpha * classif_score + (1 - alpha) * anomaly_score

        return pd.DataFrame({
            "classif_score":  classif_score,
            "anomaly_score":  anomaly_score,
            "iso_score":      comp["iso_score"].values,
            "lof_score":      comp["lof_score"].values,
            "ae_score":       comp["ae_score"].values,
            "fusion_score":   np.clip(fusion_score, 0, 100),
            "risk_tier":      pd.Series(fusion_score).apply(risk_tier).values,
            "is_anomaly":     (anomaly_score >= 50).astype(int),
            "is_flagged":     (fusion_score >= 60).astype(int),
        })

    # ── Evaluation ────────────────────────────────────────────────────────
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Evaluate anomaly detector performance using mule labels.
        Returns dict of ROC-AUC, PR-AUC, and detection rates.
        """
        scores = self.score(X)
        y_arr  = y.values.astype(int)

        roc    = roc_auc_score(y_arr, scores)
        prauc  = average_precision_score(y_arr, scores)

        # Detection rate at different thresholds
        detection = {}
        for thresh in [40, 50, 60, 70, 80]:
            preds = (scores >= thresh).astype(int)
            tp    = int(((preds == 1) & (y_arr == 1)).sum())
            fn    = int(((preds == 0) & (y_arr == 1)).sum())
            fp    = int(((preds == 1) & (y_arr == 0)).sum())
            recall = tp / (tp + fn + 1e-9)
            fpr    = fp / ((y_arr == 0).sum() + 1e-9)
            detection[f"threshold_{thresh}"] = {
                "tp": tp, "fn": fn, "fp": fp,
                "recall": round(recall, 4),
                "fpr":    round(fpr, 4),
            }

        return {
            "roc_auc":    round(roc, 4),
            "pr_auc":     round(prauc, 4),
            "detection":  detection,
            "n_mule":     int(y_arr.sum()),
            "n_legit":    int((y_arr == 0).sum()),
        }

    # ── Persistence ───────────────────────────────────────────────────────
    def save(self, out_dir: Path = AD_MODEL_DIR):
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.iso, out_dir / "isolation_forest.pkl")
        joblib.dump(self.lof, out_dir / "lof.pkl")

        if TORCH_AVAILABLE and self.ae._net is not None:
            self.ae.save_weights(str(out_dir / "autoencoder.pt"))
            ae_meta = {k: v for k, v in self.ae.__dict__.items()
                       if not k.startswith("_net") and k != "_mlp"}
            ae_meta["_n_in"] = self.ae._n_in
        else:
            ae_meta = {}

        joblib.dump(self.ae, out_dir / "autoencoder.pkl")
        joblib.dump({
            "w_iso":        self.w_iso,
            "w_lof":        self.w_lof,
            "w_ae":         self.w_ae,
            "fusion_alpha": self.fusion_alpha,
            "feature_names": self.feature_names,
        }, out_dir / "pipeline_meta.pkl")

        log.info(f"\n  Pipeline saved → {out_dir}/")

    @classmethod
    def load(cls, out_dir: Path = AD_MODEL_DIR) -> "AnomalyDetectionPipeline":
        meta     = joblib.load(out_dir / "pipeline_meta.pkl")
        pipe     = cls(
            w_iso        = meta["w_iso"],
            w_lof        = meta["w_lof"],
            w_ae         = meta["w_ae"],
            fusion_alpha = meta["fusion_alpha"],
        )
        pipe.iso           = joblib.load(out_dir / "isolation_forest.pkl")
        pipe.lof           = joblib.load(out_dir / "lof.pkl")
        pipe.ae            = joblib.load(out_dir / "autoencoder.pkl")
        pipe.feature_names = meta["feature_names"]

        if TORCH_AVAILABLE and (out_dir / "autoencoder.pt").exists():
            pipe.ae.load_weights(str(out_dir / "autoencoder.pt"))

        log.info(f"  Pipeline loaded ← {out_dir}/")
        return pipe


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — EVALUATION PLOTS
# ════════════════════════════════════════════════════════════════════════════
def plot_score_distributions(
    scores_df: pd.DataFrame,
    y: pd.Series,
    out_dir: Path = AD_PLOT_DIR,
):
    """Score distributions for legitimate vs mule accounts."""
    y_arr   = y.values
    cols    = ["iso_score", "lof_score", "ae_score", "anomaly_score"]
    titles  = ["Isolation Forest", "Local Outlier Factor",
               "Autoencoder", "Ensemble Anomaly Score"]
    colours = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Anomaly Score Distributions — Legitimate vs Mule",
                 fontsize=14, fontweight="bold")

    for ax, col, title, colour in zip(axes.flat, cols, titles, colours):
        legit_scores = scores_df.loc[y_arr == 0, col]
        mule_scores  = scores_df.loc[y_arr == 1, col]

        ax.hist(legit_scores, bins=50, color="#95a5a6", alpha=0.7,
                label=f"Legitimate (n={len(legit_scores):,})", density=True)
        ax.hist(mule_scores, bins=min(20, len(mule_scores)), color=colour,
                alpha=0.85, label=f"Mule (n={len(mule_scores):,})", density=True)

        ax.axvline(50, color="black", ls="--", lw=1.5, label="Threshold=50")
        ax.set_xlabel("Score [0–100]", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "anomaly_score_dist.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Score distributions → {out_dir}/anomaly_score_dist.png")


def plot_roc_curves(
    scores_df: pd.DataFrame,
    y: pd.Series,
    out_dir: Path = AD_PLOT_DIR,
):
    """ROC curves for each detector and the ensemble."""
    y_arr   = y.values
    cols    = ["iso_score", "lof_score", "ae_score", "anomaly_score"]
    names   = ["Isolation Forest", "LOF", "Autoencoder", "Ensemble"]
    colours = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Anomaly Detector Performance", fontsize=14, fontweight="bold")

    for col, name, colour in zip(cols, names, colours):
        lw = 2.5 if name == "Ensemble" else 1.8
        # ROC
        fpr, tpr, _ = roc_curve(y_arr, scores_df[col])
        auc         = roc_auc_score(y_arr, scores_df[col])
        axes[0].plot(fpr, tpr, color=colour, lw=lw, label=f"{name} (AUC={auc:.4f})")
        # PR
        prec, rec, _ = precision_recall_curve(y_arr, scores_df[col])
        prauc        = average_precision_score(y_arr, scores_df[col])
        axes[1].plot(rec, prec, color=colour, lw=lw, label=f"{name} (PR-AUC={prauc:.4f})")

    axes[0].plot([0,1],[0,1], "k--", lw=1.2, label="Random")
    axes[0].set_xlabel("FPR", fontsize=12); axes[0].set_ylabel("TPR", fontsize=12)
    axes[0].set_title("ROC Curve", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    axes[1].axhline(y_arr.mean(), color="gray", ls="--", lw=1.2, label="Baseline")
    axes[1].set_xlabel("Recall", fontsize=12); axes[1].set_ylabel("Precision", fontsize=12)
    axes[1].set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "anomaly_roc_pr.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  ROC/PR curves → {out_dir}/anomaly_roc_pr.png")


def plot_detector_correlation(
    scores_df: pd.DataFrame,
    out_dir: Path = AD_PLOT_DIR,
):
    """Heatmap of correlation between detector scores."""
    cols = ["iso_score", "lof_score", "ae_score", "anomaly_score"]
    corr = scores_df[cols].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    im      = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdYlGn")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    labels = ["IF", "LOF", "AE", "Ensemble"]
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)

    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if abs(corr.values[i,j]) > 0.6 else "black")

    ax.set_title("Detector Score Correlation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "detector_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Detector correlation → {out_dir}/detector_correlation.png")


def plot_fusion_analysis(
    fused_df: pd.DataFrame,
    y: pd.Series,
    out_dir: Path = AD_PLOT_DIR,
):
    """Scatter plot: classifier score vs anomaly score, coloured by true label."""
    y_arr = y.values
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Fusion Score Analysis", fontsize=14, fontweight="bold")

    # Scatter: anomaly vs classifier
    ax = axes[0]
    legit = fused_df.loc[y_arr == 0]
    mule  = fused_df.loc[y_arr == 1]
    ax.scatter(legit["classif_score"], legit["anomaly_score"],
               color="#95a5a6", alpha=0.3, s=10, label="Legitimate")
    ax.scatter(mule["classif_score"], mule["anomaly_score"],
               color="#e74c3c", alpha=0.9, s=60, marker="*", label="Mule")
    ax.axvline(50, color="#3498db", ls="--", lw=1.2, alpha=0.7, label="Classif thr=50")
    ax.axhline(50, color="#e74c3c", ls="--", lw=1.2, alpha=0.7, label="Anomaly thr=50")
    ax.set_xlabel("Classifier Score [0–100]", fontsize=12)
    ax.set_ylabel("Anomaly Score [0–100]", fontsize=12)
    ax.set_title("Classifier vs Anomaly Score", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Fusion score distribution
    ax = axes[1]
    ax.hist(fused_df.loc[y_arr==0, "fusion_score"], bins=50,
            color="#95a5a6", alpha=0.75, density=True, label="Legitimate")
    ax.hist(fused_df.loc[y_arr==1, "fusion_score"],
            bins=min(20, int((y_arr==1).sum())),
            color="#e74c3c", alpha=0.9, density=True, label="Mule")
    ax.axvline(60, color="black", ls="--", lw=1.5, label="HIGH risk thr=60")
    ax.axvline(80, color="darkred", ls="--", lw=1.5, label="CRITICAL thr=80")
    ax.set_xlabel("Fusion Score [0–100]", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Fusion Score Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "fusion_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Fusion analysis → {out_dir}/fusion_analysis.png")


def plot_reconstruction_error(
    ae: AutoencoderDetector,
    X: pd.DataFrame,
    y: pd.Series,
    out_dir: Path = AD_PLOT_DIR,
):
    """Autoencoder reconstruction error distribution."""
    X_arr  = X.values.astype(np.float32)
    errors = ae.reconstruction_error(X_arr)
    y_arr  = y.values

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Autoencoder Reconstruction Error", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.hist(errors[y_arr==0], bins=60, color="#3498db", alpha=0.7, density=True,
            label="Legitimate")
    ax.hist(errors[y_arr==1], bins=min(20, int(y_arr.sum())), color="#e74c3c",
            alpha=0.85, density=True, label="Mule")
    ax.set_xlabel("Reconstruction MSE", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Error Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.semilogy(sorted(errors[y_arr==0]), color="#3498db", alpha=0.7, label="Legitimate")
    ax.semilogy(np.sort(errors[y_arr==1]), "r*", ms=8, label="Mule")
    ax.set_xlabel("Account Rank (sorted)", fontsize=12)
    ax.set_ylabel("Reconstruction MSE (log)", fontsize=12)
    ax.set_title("Sorted Error (log scale)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "reconstruction_error.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Reconstruction error → {out_dir}/reconstruction_error.png")


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — COMBINED INFERENCE (with existing fraud_detection_model)
# ════════════════════════════════════════════════════════════════════════════
class CombinedFraudPipeline:
    """
    Combines the classification pipeline (fraud_detection_model.py) with the
    anomaly detection layer into a single, unified fraud scoring system.

    Usage
    ─────
    >>> pipeline = CombinedFraudPipeline.load()
    >>> result   = pipeline.score(df_new)
    >>> result["fusion_score"]   # [0, 100] final risk score
    >>> result["risk_tier"]      # CRITICAL / HIGH / MEDIUM / LOW
    >>> result["mule_prob"]      # raw classifier probability
    >>> result["anomaly_score"]  # raw anomaly score [0, 100]
    """

    def __init__(
        self,
        clf_pipeline,
        anomaly_pipeline: AnomalyDetectionPipeline,
        fusion_alpha: float = FUSION_ALPHA,
    ):
        self.clf      = clf_pipeline
        self.anomaly  = anomaly_pipeline
        self.alpha    = fusion_alpha

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full fraud scoring: classification + anomaly + fusion.

        Returns one row per input account with:
          mule_prob, classif_score, anomaly_score, iso_score, lof_score,
          ae_score, fusion_score, risk_tier, is_flagged,
          risk_label (from classifier alone)
        """
        # Classification predictions
        clf_result = self.clf.predict(df)
        mule_prob  = clf_result["mule_prob"].values

        # Anomaly + fusion
        fused = self.anomaly.fuse(df, mule_prob, alpha=self.alpha)

        # Merge
        result = pd.DataFrame({
            "mule_prob":      mule_prob,
            "risk_label":     clf_result["risk_label"].values,
            "classif_score":  fused["classif_score"].values,
            "anomaly_score":  fused["anomaly_score"].values,
            "iso_score":      fused["iso_score"].values,
            "lof_score":      fused["lof_score"].values,
            "ae_score":       fused["ae_score"].values,
            "fusion_score":   fused["fusion_score"].values,
            "risk_tier":      fused["risk_tier"].values,
            "is_anomaly":     fused["is_anomaly"].values,
            "is_flagged":     fused["is_flagged"].values,
            "flag_count":     clf_result.get("flag_count", pd.Series([0]*len(df))).values,
        })

        return result

    def save(self, path: str = "anomaly_detection/models/combined_pipeline.pkl"):
        joblib.dump(self, path)
        log.info(f"  Combined pipeline saved → {path}")

    @classmethod
    def load(
        cls,
        clf_path: str   = "models/inference_pipeline.pkl",
        ad_dir: Path    = AD_MODEL_DIR,
        fusion_alpha: float = FUSION_ALPHA,
    ) -> "CombinedFraudPipeline":
        try:
            sys.path.insert(0, str(BASE_DIR))
            from fraud_detection_model import MuleAccountInferencePipeline
            clf = MuleAccountInferencePipeline.load(clf_path)
        except Exception as e:
            log.error(f"Could not load classifier pipeline: {e}")
            clf = None

        anomaly = AnomalyDetectionPipeline.load(ad_dir)
        return cls(clf, anomaly, fusion_alpha)


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    start = datetime.now()
    log.info("=" * 60)
    log.info("  ANOMALY DETECTION LAYER — TRAINING")
    log.info(f"  Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────
    X_legit, X_all, y = load_data()

    # ── Train pipeline ────────────────────────────────────────────────────
    ad_pipeline = AnomalyDetectionPipeline(
        w_iso        = W_ISO,
        w_lof        = W_LOF,
        w_ae         = W_AE,
        fusion_alpha = FUSION_ALPHA,
        ae_params    = {"n_epochs": 100, "batch_size": 64, "patience": 15},
        if_params    = {"n_estimators": 300, "contamination": 0.01},
        lof_params   = {"n_neighbors": 20, "contamination": 0.01},
    )
    ad_pipeline.fit(X_legit)

    # ── Score all data ────────────────────────────────────────────────────
    log.info("\n  Scoring all accounts …")
    scores_df = ad_pipeline.score_components(X_all)
    scores_df["anomaly_tier"] = scores_df["anomaly_score"].apply(risk_tier)

    # ── Evaluation ────────────────────────────────────────────────────────
    log.info("\n  Evaluating anomaly detector performance …")
    eval_results = ad_pipeline.evaluate(X_all, y)

    log.info("\n" + "─"*60)
    log.info("  ANOMALY DETECTOR EVALUATION")
    log.info("─"*60)
    log.info(f"  ROC-AUC  : {eval_results['roc_auc']:.4f}")
    log.info(f"  PR-AUC   : {eval_results['pr_auc']:.4f}")
    log.info("\n  Detection rates by threshold:")
    log.info(f"  {'Threshold':>10} {'Recall':>9} {'FPR':>7} {'TP':>5} {'FP':>7}")
    log.info("  " + "─"*40)
    for thr, det in eval_results["detection"].items():
        thr_val = thr.split("_")[1]
        log.info(f"  {thr_val:>10}   {det['recall']:>7.4f}  "
                 f"{det['fpr']:>7.4f}  {det['tp']:>5}  {det['fp']:>7}")
    log.info("─"*60)

    # ── Try fusion with classifier ────────────────────────────────────────
    fused_df = None
    try:
        sys.path.insert(0, str(BASE_DIR))
        from fraud_detection_model import MuleAccountInferencePipeline
        clf = MuleAccountInferencePipeline.load("models/inference_pipeline.pkl")
        log.info("\n  Fusing with classifier probabilities …")
        clf_result = clf.predict(X_all)
        mule_prob  = clf_result["mule_prob"].values
        fused_df   = ad_pipeline.fuse(X_all, mule_prob)
        log.info("  Fusion complete.")
    except Exception as e:
        log.warning(f"  Could not load classifier pipeline: {e}")
        log.warning("  Fusion plot skipped. Run fraud_detection_model.py first.")

    # ── Generate plots ────────────────────────────────────────────────────
    log.info("\n  Generating plots …")
    plot_score_distributions(scores_df, y)
    plot_roc_curves(scores_df, y)
    plot_detector_correlation(scores_df)
    plot_reconstruction_error(ad_pipeline.ae, X_all, y)
    if fused_df is not None:
        plot_fusion_analysis(fused_df, y)

    # ── Save pipeline ─────────────────────────────────────────────────────
    log.info("\n  Saving anomaly detection pipeline …")
    ad_pipeline.save()

    # Save evaluation results
    with open(AD_MODEL_DIR / "evaluation_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    # Score summary
    log.info("\n  Score summary:")
    for label, mask in [("Legitimate", y == 0), ("Mule", y == 1)]:
        s = scores_df.loc[mask.values, "anomaly_score"]
        log.info(f"  {label:<12}  mean={s.mean():.1f}  "
                 f"p50={s.median():.1f}  p90={s.quantile(0.9):.1f}  "
                 f"p99={s.quantile(0.99):.1f}")

    # ── Try building combined pipeline ────────────────────────────────────
    if fused_df is not None:
        try:
            combined = CombinedFraudPipeline(clf, ad_pipeline, FUSION_ALPHA)
            combined.save("anomaly_detection/models/combined_pipeline.pkl")
            log.info("  Combined pipeline saved.")
        except Exception as e:
            log.warning(f"  Could not save combined pipeline: {e}")

    elapsed = (datetime.now() - start).total_seconds()
    log.info(f"\n✅  Done in {elapsed/60:.1f} minutes")
    log.info(f"\n  Output files:")
    for f in sorted(AD_MODEL_DIR.iterdir()):
        log.info(f"    {f}")
    for f in sorted(AD_PLOT_DIR.iterdir()):
        log.info(f"    {f}")

    # ── Demo inference ────────────────────────────────────────────────────
    log.info("\n  Demo: anomaly scores for 5 sample accounts …")
    sample = X_all.sample(5, random_state=42)
    sample_y = y.iloc[sample.index]
    pred = ad_pipeline.predict(sample)
    pred["true_label"] = sample_y.values
    log.info("\n" + pred[["anomaly_score", "iso_score", "lof_score",
                           "ae_score", "anomaly_tier", "true_label"]].to_string())


if __name__ == "__main__":
    main()
