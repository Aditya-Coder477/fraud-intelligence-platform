"""
================================================================================
  BANKING RISK SCORING ENGINE
  Version : 1.0.0
================================================================================

ARCHITECTURE
────────────
  Inputs (all normalised to [0, 100])
    ├── ml_prob          : ML fraud probability score  (weight 0.40)
    ├── anomaly_score    : Anomaly detection score     (weight 0.25)
    ├── behavior_score   : Transaction behaviour score  (weight 0.20)
    └── alert_score      : Rule-based alert score      (weight 0.15)
                                │
                    ┌───────────┴───────────┐
                    │                       │
              Base Score              Boost Engine
              (weighted avg)          ─ threshold boosts
                    │                 ─ interaction boosts
                    └───────────┬───────────┘
                                │
                         Final Score
                          clip(base × boost, 0, 100)
                                │
                  ┌─────────────┴──────────────┐
                  │  Risk Categories           │
                  │   0–30  : SAFE             │
                  │  31–60  : MONITOR          │
                  │  61–80  : REVIEW           │
                  │  81–100 : BLOCK            │
                  └────────────────────────────┘
                                │
                   Explainability Module
                    ─ component contributions
                    ─ boost attribution
                    ─ natural language summary
                    ─ waterfall chart

SCORING FORMULA (detailed)
──────────────────────────
  base_score = 0.40×ml + 0.25×anomaly + 0.20×behavior + 0.15×alert

  boost = 1.0
    + HIGH_SIGNAL_BOOSTS  (per-component, if score > HIGH_THRESH=75)
    + CRITICAL_ALERT_BOOST (if alert_score > CRITICAL_THRESH=85)
    + INTERACTION_BOOSTS  (if ≥2 or ≥3 signals exceed ELEVATED_THRESH=60)

  final_score = clip(base_score × boost, 0, 100)

RISK CATEGORIES
───────────────
   0–30  SAFE      → Standard processing, no action
  31–60  MONITOR   → Enhanced monitoring, flag for review
  61–80  REVIEW    → Hold for analyst review, delay transaction
  81–100 BLOCK     → Immediate block, SAR filing, escalation
"""

# ── stdlib ───────────────────────────────────────────────────────────────────
import json
import math
import logging
import warnings
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("RiskEngine")

# ── third-party ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 0 — CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

# ── Component weights (must sum to 1.0) ─────────────────────────────────────
WEIGHTS = {
    "ml_prob":       0.40,   # most reliable — trained on historical fraud
    "anomaly_score": 0.25,   # unsupervised signal — catches novel patterns
    "behavior_score":0.20,   # transaction behaviour — velocity, amount patterns
    "alert_score":   0.15,   # rule-based alerts — regulatory / policy triggers
}

# ── Boost thresholds ─────────────────────────────────────────────────────────
HIGH_THRESH      = 75.0   # individual component boost trigger
CRITICAL_THRESH  = 85.0   # critical alert boost trigger
ELEVATED_THRESH  = 60.0   # interaction: signal is "elevated"

# ── Per-component boost amounts (added to boost multiplier) ──────────────────
HIGH_SIGNAL_BOOST = {
    "ml_prob":        0.12,
    "anomaly_score":  0.10,
    "behavior_score": 0.08,
    "alert_score":    0.15,
}
CRITICAL_ALERT_BOOST   = 0.20  # extra boost if alert_score > CRITICAL_THRESH
INTERACTION_BOOST_2    = 0.10  # ≥2 signals elevated
INTERACTION_BOOST_3    = 0.20  # ≥3 signals elevated (cumulative on top of 2)

# ── Risk categories ──────────────────────────────────────────────────────────
RISK_CATEGORIES = [
    (81, 100, "BLOCK",   "#c0392b", "Immediate block, SAR filing, escalation"),
    (61, 80,  "REVIEW",  "#e67e22", "Hold for analyst review, delay transaction"),
    (31, 60,  "MONITOR", "#f1c40f", "Enhanced monitoring, flag for review queue"),
    (0,  30,  "SAFE",    "#27ae60", "Standard processing, no action required"),
]

OUTPUT_DIR = Path("risk_engine")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "plots").mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class RiskInput:
    """
    Normalised input to the risk scoring engine.
    All scores must be in [0, 100].
    """
    ml_prob:        float    # ML fraud probability × 100
    anomaly_score:  float    # Anomaly detection score [0, 100]
    behavior_score: float    # Transaction behaviour score [0, 100]
    alert_score:    float    # Rule-based alert score [0, 100]

    # Optional metadata
    account_id:     str      = ""
    timestamp:      str      = field(default_factory=lambda: datetime.now().isoformat())
    transaction_id: str      = ""
    channel:        str      = ""

    def validate(self):
        for attr in ("ml_prob", "anomaly_score", "behavior_score", "alert_score"):
            v = getattr(self, attr)
            if not (0.0 <= v <= 100.0):
                raise ValueError(
                    f"'{attr}' must be in [0, 100], got {v}. "
                    f"If passing probabilities, multiply by 100 first."
                )
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RiskInput":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    @classmethod
    def from_prob(
        cls,
        ml_prob_raw: float,           # raw probability [0, 1]
        anomaly_score: float,
        behavior_score: float,
        alert_score: float,
        **kwargs,
    ) -> "RiskInput":
        """Convenience constructor: accepts ml_prob as raw probability [0, 1]."""
        return cls(
            ml_prob        = ml_prob_raw * 100,
            anomaly_score  = anomaly_score,
            behavior_score = behavior_score,
            alert_score    = alert_score,
            **kwargs,
        ).validate()


@dataclass
class BoostTrace:
    """Tracks every boost applied during scoring."""
    base_score:          float
    boost_multiplier:    float
    high_signal_boosts:  dict = field(default_factory=dict)
    critical_alert_boost:float = 0.0
    interaction_boost:   float = 0.0
    n_elevated:          int   = 0
    final_score:         float = 0.0

    def total_boost_added(self) -> float:
        return self.boost_multiplier - 1.0

    def boost_breakdown(self) -> dict:
        return {
            "high_signal": self.high_signal_boosts,
            "critical_alert": self.critical_alert_boost,
            "interaction": self.interaction_boost,
            "total_multiplier": self.boost_multiplier,
        }


@dataclass
class RiskOutput:
    """Complete risk assessment output."""
    score:          float
    category:       str
    category_color: str
    category_action:str
    weights:        dict
    components:     dict     # weighted contribution of each input
    boost:          BoostTrace
    inputs:         dict
    account_id:     str      = ""
    timestamp:      str      = field(default_factory=lambda: datetime.now().isoformat())
    explanation:    dict     = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def summary(self) -> str:
        lines = [
            f"╔══════════════════════════════════════════════════╗",
            f"║  RISK ASSESSMENT REPORT                          ║",
            f"╠══════════════════════════════════════════════════╣",
            f"║  Account      : {self.account_id:<32} ║",
            f"║  Timestamp    : {self.timestamp[:19]:<32} ║",
            f"╠══════════════════════════════════════════════════╣",
            f"║  RISK SCORE   : {self.score:>5.1f} / 100                     ║",
            f"║  CATEGORY     : {self.category:<32} ║",
            f"║  ACTION       : {self.category_action:<32} ║",
            f"╠══════════════════════════════════════════════════╣",
            f"║  Component Scores                                ║",
            f"║  ML Fraud Prob  : {self.inputs['ml_prob']:>5.1f}  →  "
            f"contrib: {self.components['ml_prob']:>5.1f}           ║",
            f"║  Anomaly Score  : {self.inputs['anomaly_score']:>5.1f}  →  "
            f"contrib: {self.components['anomaly_score']:>5.1f}           ║",
            f"║  Behavior Score : {self.inputs['behavior_score']:>5.1f}  →  "
            f"contrib: {self.components['behavior_score']:>5.1f}           ║",
            f"║  Alert Score    : {self.inputs['alert_score']:>5.1f}  →  "
            f"contrib: {self.components['alert_score']:>5.1f}           ║",
            f"╠══════════════════════════════════════════════════╣",
            f"║  Base Score   : {self.boost.base_score:>5.1f}                        ║",
            f"║  Boost ×      : {self.boost.boost_multiplier:>5.3f}  "
            f"(+{self.boost.total_boost_added():.3f})              ║",
            f"║  N Elevated   : {self.boost.n_elevated}                              ║",
            f"╚══════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — SCORING FORMULA ENGINE
# ════════════════════════════════════════════════════════════════════════════
class ScoringFormula:
    """
    Core mathematical scoring formula.

    Formula
    ───────
    base_score = Σ(wᵢ × scoreᵢ)

    boost = 1.0
      + Σ HIGH_SIGNAL_BOOST[i]  if  scoreᵢ > HIGH_THRESH          (per component)
      + CRITICAL_ALERT_BOOST    if  alert_score > CRITICAL_THRESH
      + INTERACTION_BOOST_2     if  n_elevated ≥ 2
      + INTERACTION_BOOST_3     if  n_elevated ≥ 3   (additional)

    final_score = clip(base_score × boost, 0, 100)

    Rationale
    ─────────
    The boost mechanism is critical for AML:
    - A single high signal (e.g. high ML prob alone) amplifies the score → prevents
      high-risk accounts from escaping because one signal is suppressed.
    - Multiple elevated signals multiply the boost → reflects the convergent evidence
      principle: an account flagged by 3 independent methods is exponentially more
      suspicious than one flagged by 1.
    """

    def __init__(
        self,
        weights:              dict  = None,
        high_thresh:          float = HIGH_THRESH,
        critical_thresh:      float = CRITICAL_THRESH,
        elevated_thresh:      float = ELEVATED_THRESH,
        high_signal_boost:    dict  = None,
        critical_alert_boost: float = CRITICAL_ALERT_BOOST,
        interaction_boost_2:  float = INTERACTION_BOOST_2,
        interaction_boost_3:  float = INTERACTION_BOOST_3,
    ):
        self.weights              = weights or deepcopy(WEIGHTS)
        self.high_thresh          = high_thresh
        self.critical_thresh      = critical_thresh
        self.elevated_thresh      = elevated_thresh
        self.high_signal_boost    = high_signal_boost or deepcopy(HIGH_SIGNAL_BOOST)
        self.critical_alert_boost = critical_alert_boost
        self.interaction_boost_2  = interaction_boost_2
        self.interaction_boost_3  = interaction_boost_3

        # Validate weights
        total = sum(self.weights.values())
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total:.4f}")

    def compute(self, inp: RiskInput) -> RiskOutput:
        """Compute risk score with full trace."""
        scores = {
            "ml_prob":        inp.ml_prob,
            "anomaly_score":  inp.anomaly_score,
            "behavior_score": inp.behavior_score,
            "alert_score":    inp.alert_score,
        }

        # ── Step 1: Base weighted score ───────────────────────────────────
        components   = {k: self.weights[k] * v for k, v in scores.items()}
        base_score   = sum(components.values())

        # ── Step 2: Boost calculation ─────────────────────────────────────
        boost        = 1.0
        high_boosts  = {}

        # Per-component high-signal boost
        for k, v in scores.items():
            if v > self.high_thresh:
                b = self.high_signal_boost[k]
                high_boosts[k] = b
                boost          += b

        # Critical alert override
        crit_boost = 0.0
        if scores["alert_score"] > self.critical_thresh:
            crit_boost = self.critical_alert_boost
            boost      += crit_boost

        # Interaction boost: how many signals are elevated (>60)?
        n_elevated  = sum(1 for v in scores.values() if v > self.elevated_thresh)
        inter_boost = 0.0
        if n_elevated >= 3:
            inter_boost = self.interaction_boost_2 + self.interaction_boost_3
            boost       += inter_boost
        elif n_elevated >= 2:
            inter_boost = self.interaction_boost_2
            boost       += inter_boost

        # ── Step 3: Final score ───────────────────────────────────────────
        final_score = float(np.clip(base_score * boost, 0.0, 100.0))

        # ── Step 4: Category assignment ───────────────────────────────────
        category, colour, action = "SAFE", "#27ae60", "Standard processing"
        for lo, hi, cat, col, act in RISK_CATEGORIES:
            if lo <= final_score <= hi:
                category, colour, action = cat, col, act
                break

        boost_trace = BoostTrace(
            base_score           = round(base_score, 4),
            boost_multiplier     = round(boost, 4),
            high_signal_boosts   = {k: round(v, 4) for k, v in high_boosts.items()},
            critical_alert_boost = round(crit_boost, 4),
            interaction_boost    = round(inter_boost, 4),
            n_elevated           = n_elevated,
            final_score          = round(final_score, 2),
        )

        return RiskOutput(
            score          = round(final_score, 2),
            category       = category,
            category_color = colour,
            category_action= action,
            weights        = dict(self.weights),
            components     = {k: round(v, 4) for k, v in components.items()},
            boost          = boost_trace,
            inputs         = {k: round(v, 2) for k, v in scores.items()},
            account_id     = inp.account_id,
            timestamp      = inp.timestamp,
        )

    def batch_compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute risk scores for a DataFrame of inputs.

        Expected columns: ml_prob, anomaly_score, behavior_score, alert_score
        Optional columns: account_id, transaction_id, channel

        Returns DataFrame with all scores and categories added.
        """
        results = []
        for _, row in df.iterrows():
            inp = RiskInput(
                ml_prob        = float(row.get("ml_prob", 0)),
                anomaly_score  = float(row.get("anomaly_score", 0)),
                behavior_score = float(row.get("behavior_score", 0)),
                alert_score    = float(row.get("alert_score", 0)),
                account_id     = str(row.get("account_id", "")),
                transaction_id = str(row.get("transaction_id", "")),
                channel        = str(row.get("channel", "")),
            ).validate()
            out = self.compute(inp)
            results.append({
                "account_id":        inp.account_id,
                "risk_score":        out.score,
                "risk_category":     out.category,
                "ml_contrib":        out.components["ml_prob"],
                "anomaly_contrib":   out.components["anomaly_score"],
                "behavior_contrib":  out.components["behavior_score"],
                "alert_contrib":     out.components["alert_score"],
                "base_score":        out.boost.base_score,
                "boost_multiplier":  out.boost.boost_multiplier,
                "n_elevated":        out.boost.n_elevated,
            })

        return pd.DataFrame(results)


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — EXPLAINABILITY MODULE
# ════════════════════════════════════════════════════════════════════════════
class RiskExplainer:
    """
    Generates human-readable explanations for every risk assessment.

    Explanation layers:
      1. Component contributions    (waterfall chart)
      2. Boost attribution          (which rules fired)
      3. Natural language summary   (regulatory-grade prose)
      4. Counter-factual            (what would lower the score)
    """

    COMPONENT_LABELS = {
        "ml_prob":        "ML Fraud Probability",
        "anomaly_score":  "Anomaly Detection",
        "behavior_score": "Transaction Behaviour",
        "alert_score":    "Rule-Based Alerts",
    }

    COMPONENT_DESCRIPTIONS = {
        "ml_prob": (
            "Gradient-boosted ensemble model trained on historical mule/fraud patterns. "
            "Considers 155 engineered features including transaction flows, behavioural "
            "patterns, and account demographics."
        ),
        "anomaly_score": (
            "Isolation Forest + LOF + Autoencoder ensemble. Measures deviation from the "
            "normal distribution of legitimate accounts. Catches novel fraud patterns "
            "not present in historical training data."
        ),
        "behavior_score": (
            "Transaction velocity, amount distribution, time-of-day patterns, "
            "counterparty concentration, and flow balance metrics. Core AML signal: "
            "near-zero net balance (in = out) indicates pass-through behaviour."
        ),
        "alert_score": (
            "Rule-based alerts from the AML policy engine: high-value transactions, "
            "round-dollar amounts, PEP matches, sanction list hits, rapid succession "
            "transfers, and geographical risk indicators."
        ),
    }

    def explain(self, output: RiskOutput) -> dict:
        """
        Generate full structured explanation for a risk assessment.

        Returns dict with:
          components, boosts, narrative, counterfactual, severity_drivers
        """
        explanation = {
            "components":         self._explain_components(output),
            "boosts":             self._explain_boosts(output),
            "narrative":          self._generate_narrative(output),
            "counterfactual":     self._generate_counterfactual(output),
            "severity_drivers":   self._identify_severity_drivers(output),
            "dominant_signal":    self._dominant_signal(output),
        }
        return explanation

    def _explain_components(self, output: RiskOutput) -> list:
        items = []
        for key, contrib in output.components.items():
            raw_score = output.inputs[key]
            pct       = (contrib / output.score * 100) if output.score > 0 else 0
            items.append({
                "component":    key,
                "label":        self.COMPONENT_LABELS[key],
                "raw_score":    raw_score,
                "weight":       output.weights[key],
                "contribution": contrib,
                "pct_of_total": round(pct, 1),
                "level":        self._level_label(raw_score),
                "description":  self.COMPONENT_DESCRIPTIONS[key],
            })
        return sorted(items, key=lambda x: x["contribution"], reverse=True)

    def _explain_boosts(self, output: RiskOutput) -> dict:
        bt = output.boost
        items = []

        for comp, boost_val in bt.high_signal_boosts.items():
            items.append({
                "rule":        f"HIGH_SIGNAL: {self.COMPONENT_LABELS[comp]}",
                "condition":   f"{self.COMPONENT_LABELS[comp]} > {HIGH_THRESH}",
                "value":       f"{output.inputs[comp]:.1f}",
                "boost_added": f"+{boost_val:.3f}×",
                "fired":       True,
            })

        if bt.critical_alert_boost > 0:
            items.append({
                "rule":        "CRITICAL_ALERT: Rule-Based Alert",
                "condition":   f"Alert Score > {CRITICAL_THRESH}",
                "value":       f"{output.inputs['alert_score']:.1f}",
                "boost_added": f"+{bt.critical_alert_boost:.3f}×",
                "fired":       True,
            })

        if bt.interaction_boost > 0:
            items.append({
                "rule":        f"CONVERGENT_EVIDENCE: {bt.n_elevated} signals elevated",
                "condition":   f"≥{bt.n_elevated} scores > {ELEVATED_THRESH}",
                "value":       f"n={bt.n_elevated}",
                "boost_added": f"+{bt.interaction_boost:.3f}×",
                "fired":       True,
            })

        # List rules that did NOT fire
        if not bt.high_signal_boosts:
            items.append({
                "rule":        "HIGH_SIGNAL: No component exceeded threshold",
                "condition":   f"Any score > {HIGH_THRESH}",
                "value":       "N/A",
                "boost_added": "+0.000×",
                "fired":       False,
            })

        return {
            "base_score":       bt.base_score,
            "boost_multiplier": bt.boost_multiplier,
            "boost_added":      round(bt.total_boost_added(), 4),
            "rules":            items,
            "n_elevated":       bt.n_elevated,
        }

    def _generate_narrative(self, output: RiskOutput) -> str:
        """Regulatory-grade natural language explanation."""
        cat   = output.category
        score = output.score
        bt    = output.boost
        inp   = output.inputs
        dom   = self._dominant_signal(output)

        # Opening
        cat_text = {
            "BLOCK":   "presents critically elevated fraud risk indicators",
            "REVIEW":  "exhibits significant fraud risk indicators requiring analyst review",
            "MONITOR": "shows moderate fraud risk indicators warranting enhanced monitoring",
            "SAFE":    "shows low fraud risk indicators consistent with normal behaviour",
        }
        narrative = (
            f"This account {cat_text[cat]} with a composite risk score of "
            f"{score:.1f}/100 (category: {cat}). "
        )

        # Primary driver
        narrative += (
            f"The primary risk driver is the {self.COMPONENT_LABELS[dom['component']]} "
            f"({dom['component'].replace('_', ' ')}: {inp[dom['component']]:.1f}/100), "
            f"which contributes {dom['contribution']:.1f} points ({dom['pct_of_total']:.0f}%) "
            f"to the final score. "
        )

        # Component breakdown
        elevated = [
            f"{self.COMPONENT_LABELS[k]} ({inp[k]:.0f})"
            for k in inp if inp[k] >= ELEVATED_THRESH
        ]
        if elevated:
            narrative += (
                f"The following signals are elevated (≥{ELEVATED_THRESH:.0f}): "
                f"{', '.join(elevated)}. "
            )

        # Boost explanation
        if bt.total_boost_added() > 0.01:
            narrative += (
                f"The base score of {bt.base_score:.1f} was amplified by a boost "
                f"multiplier of {bt.boost_multiplier:.3f}× (total boost: "
                f"+{bt.total_boost_added()*100:.0f}%) due to: "
            )
            reasons = []
            if bt.high_signal_boosts:
                comps = [self.COMPONENT_LABELS[k] for k in bt.high_signal_boosts]
                reasons.append(
                    f"high-signal thresholds exceeded in {', '.join(comps)}"
                )
            if bt.critical_alert_boost > 0:
                reasons.append("critical alert threshold exceeded")
            if bt.interaction_boost > 0:
                reasons.append(
                    f"convergent evidence from {bt.n_elevated} independent signals"
                )
            narrative += "; ".join(reasons) + ". "
        else:
            narrative += "No signal amplification boosts were triggered. "

        # Action
        narrative += f"Recommended action: {output.category_action}."
        return narrative

    def _generate_counterfactual(self, output: RiskOutput) -> dict:
        """What changes would move the account to a lower risk category?"""
        score    = output.score
        cat      = output.category

        if cat == "SAFE":
            return {"message": "Account is already in the lowest risk category (SAFE)."}

        # Target: drop one category
        target_cats = {
            "BLOCK":   ("REVIEW",  80),
            "REVIEW":  ("MONITOR", 60),
            "MONITOR": ("SAFE",    30),
        }
        target_cat, target_score = target_cats[cat]
        reduction_needed = score - target_score

        # Which component, if reduced, would achieve this?
        suggestions = []
        for k, w in WEIGHTS.items():
            effective_w = w * output.boost.boost_multiplier
            raw         = output.inputs[k]
            # How much would raw need to drop to reduce final by reduction_needed?
            needed_drop = reduction_needed / effective_w if effective_w > 0 else float("inf")
            new_raw     = max(0, raw - needed_drop)
            if needed_drop <= raw:
                suggestions.append({
                    "component":    k,
                    "label":        self.COMPONENT_LABELS[k],
                    "current":      round(raw, 1),
                    "needed":       round(new_raw, 1),
                    "drop_needed":  round(needed_drop, 1),
                })

        suggestions = sorted(suggestions, key=lambda x: x["drop_needed"])

        return {
            "current_category":  cat,
            "target_category":   target_cat,
            "current_score":     score,
            "target_score":      target_score,
            "reduction_needed":  round(reduction_needed, 1),
            "suggestions":       suggestions[:2],  # top 2 most achievable
            "note": (
                f"Reducing {suggestions[0]['label']} from "
                f"{suggestions[0]['current']:.0f} to "
                f"{suggestions[0]['needed']:.0f} would be sufficient "
                f"to move to {target_cat} category, assuming other scores unchanged."
            ) if suggestions else "",
        }

    def _identify_severity_drivers(self, output: RiskOutput) -> list:
        """
        Return ordered list of factors driving severity,
        including non-linear boosts.
        """
        drivers = []
        for k, contrib in output.components.items():
            raw = output.inputs[k]
            drivers.append({
                "factor": self.COMPONENT_LABELS[k],
                "impact": "HIGH" if raw >= HIGH_THRESH else
                          "MEDIUM" if raw >= ELEVATED_THRESH else "LOW",
                "raw_score":    raw,
                "contribution": contrib,
            })

        # Add boost drivers
        if output.boost.n_elevated >= 2:
            drivers.append({
                "factor":     f"Convergent Evidence ({output.boost.n_elevated} signals elevated)",
                "impact":     "HIGH",
                "raw_score":  output.boost.n_elevated,
                "contribution": output.boost.interaction_boost * output.boost.base_score,
            })
        if output.boost.critical_alert_boost > 0:
            drivers.append({
                "factor":     "Critical Alert Override",
                "impact":     "CRITICAL",
                "raw_score":  output.inputs["alert_score"],
                "contribution": output.boost.critical_alert_boost * output.boost.base_score,
            })

        return sorted(drivers, key=lambda x: x["contribution"], reverse=True)

    @staticmethod
    def _dominant_signal(output: RiskOutput) -> dict:
        comps = [
            {
                "component":    k,
                "label":        RiskExplainer.COMPONENT_LABELS[k],
                "raw_score":    output.inputs[k],
                "contribution": v,
                "pct_of_total": round(v / output.score * 100, 1) if output.score > 0 else 0,
            }
            for k, v in output.components.items()
        ]
        return max(comps, key=lambda x: x["contribution"])

    @staticmethod
    def _level_label(score: float) -> str:
        if score >= HIGH_THRESH:   return "HIGH"
        if score >= ELEVATED_THRESH: return "ELEVATED"
        if score >= 30:            return "MODERATE"
        return "LOW"


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — VISUALISATION MODULE
# ════════════════════════════════════════════════════════════════════════════
class RiskVisualiser:
    """Generates production-quality risk score visualisations."""

    COMPONENT_COLOURS = {
        "ml_prob":        "#3498db",
        "anomaly_score":  "#9b59b6",
        "behavior_score": "#2ecc71",
        "alert_score":    "#e74c3c",
    }
    BOOST_COLOUR    = "#e67e22"
    CATEGORY_COLOURS = {
        "SAFE": "#27ae60", "MONITOR": "#f39c12",
        "REVIEW": "#e67e22", "BLOCK": "#c0392b",
    }

    def waterfall_chart(
        self,
        output: RiskOutput,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Waterfall chart: how each component and boost contributes to the
        final risk score, starting from 0 and building up step by step.
        """
        components = output.components
        bt         = output.boost

        # Build waterfall steps
        steps = []
        running = 0.0
        for k, contrib in components.items():
            steps.append({
                "label":  self.COMPONENT_LABELS_SHORT[k],
                "value":  contrib,
                "start":  running,
                "end":    running + contrib,
                "color":  self.COMPONENT_COLOURS[k],
                "type":   "component",
            })
            running += contrib

        # Boost step = final - base
        boost_contribution = output.score - bt.base_score
        if boost_contribution > 0.01:
            steps.append({
                "label":  f"Boost ×{bt.boost_multiplier:.3f}",
                "value":  boost_contribution,
                "start":  running,
                "end":    running + boost_contribution,
                "color":  self.BOOST_COLOUR,
                "type":   "boost",
            })

        # Final score bar
        cat_col = self.CATEGORY_COLOURS.get(output.category, "#555")
        steps.append({
            "label":  f"FINAL\n{output.score:.1f}",
            "value":  output.score,
            "start":  0,
            "end":    output.score,
            "color":  cat_col,
            "type":   "total",
        })

        fig, ax = plt.subplots(figsize=(12, 6))
        y_pos   = range(len(steps))

        for i, s in enumerate(steps):
            if s["type"] == "total":
                ax.barh(i, s["value"], left=0, color=s["color"], alpha=0.9,
                        height=0.55, edgecolor="white", linewidth=1.5)
                ax.text(s["value"] + 0.5, i, f"  {s['value']:.1f}",
                        va="center", fontsize=11, fontweight="bold", color=s["color"])
            else:
                ax.barh(i, s["value"], left=s["start"], color=s["color"], alpha=0.85,
                        height=0.55, edgecolor="white", linewidth=1.0)
                ax.text(s["end"] + 0.5, i, f"  +{s['value']:.1f}",
                        va="center", fontsize=9, color=s["color"])

        # Risk zone shading
        for lo, hi, cat, col, _ in RISK_CATEGORIES:
            ax.axvspan(lo, hi, alpha=0.04, color=col, zorder=0)
        for val, label in [(30, "SAFE|MONITOR"), (60, "MONITOR|REVIEW"), (80, "REVIEW|BLOCK")]:
            ax.axvline(val, color="gray", ls="--", lw=0.8, alpha=0.5, zorder=1)
            ax.text(val, len(steps) - 0.3, f" {val}", fontsize=7, color="gray", va="top")

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels([s["label"] for s in steps], fontsize=10)
        ax.set_xlim(0, 105)
        ax.set_xlabel("Score Contribution [0–100]", fontsize=12)
        ax.set_title(
            f"Risk Score Waterfall — Account: {output.account_id or 'N/A'}  "
            f"│  Final: {output.score:.1f} [{output.category}]",
            fontsize=13, fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    COMPONENT_LABELS_SHORT = {
        "ml_prob":        "ML Fraud Prob",
        "anomaly_score":  "Anomaly",
        "behavior_score": "Behaviour",
        "alert_score":    "Alert",
    }

    def gauge_chart(
        self,
        output: RiskOutput,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Semi-circular gauge showing final risk score."""
        score  = output.score
        cat    = output.category
        colour = self.CATEGORY_COLOURS.get(cat, "#555")

        fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={"projection": "polar"})
        # Draw zone arcs
        zone_colours = ["#27ae60", "#f39c12", "#e67e22", "#c0392b"]
        zone_bounds  = [0, 30, 60, 80, 100]
        for i, (lo, hi, col) in enumerate(
            zip(zone_bounds[:-1], zone_bounds[1:], zone_colours)
        ):
            theta_lo = math.pi - (lo / 100) * math.pi
            theta_hi = math.pi - (hi / 100) * math.pi
            theta    = np.linspace(theta_lo, theta_hi, 100)
            ax.bar(
                x=theta.mean(), height=0.4, width=theta_lo - theta_hi,
                bottom=0.6, color=col, alpha=0.7,
            )

        # Needle
        theta_needle = math.pi - (score / 100) * math.pi
        ax.annotate(
            "", xy=(theta_needle, 0.95),
            xytext=(math.pi / 2, 0),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=2.5),
        )

        ax.set_theta_zero_location("W")
        ax.set_theta_direction(1)
        ax.set_ylim(0, 1.1)
        ax.set_xticks([math.pi, math.pi * 0.7, math.pi * 0.4, math.pi * 0.2, 0])
        ax.set_xticklabels(["0\nSAFE", "30", "60\nREVIEW", "80", "100\nBLOCK"], fontsize=9)
        ax.set_yticks([])
        ax.spines["polar"].set_visible(False)
        ax.set_facecolor("#f8f9fa")

        ax.text(
            math.pi / 2, 0.25,
            f"{score:.0f}",
            ha="center", va="center", fontsize=40, fontweight="bold", color=colour,
            transform=ax.transData,
        )
        ax.set_title(f"Risk Score Gauge — {cat}", fontsize=13, fontweight="bold", pad=20)

        fig.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def component_radar(
        self,
        output: RiskOutput,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Radar/spider chart of all four input scores."""
        labels = list(self.COMPONENT_LABELS_SHORT.values())
        vals   = [
            output.inputs["ml_prob"],
            output.inputs["anomaly_score"],
            output.inputs["behavior_score"],
            output.inputs["alert_score"],
        ]
        N     = len(labels)
        angles= [n / float(N) * 2 * math.pi for n in range(N)] + [0]
        vals_plot = vals + [vals[0]]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=11, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(["25", "50", "75", "100"], size=8)

        # Zone fills
        for lo, hi, cat, col, _ in RISK_CATEGORIES:
            ax.fill_between(angles, lo, hi, alpha=0.06, color=col)

        # Data
        ax.plot(angles, vals_plot, linewidth=2.5,
                color=self.CATEGORY_COLOURS.get(output.category, "#3498db"))
        ax.fill(angles, vals_plot,
                color=self.CATEGORY_COLOURS.get(output.category, "#3498db"), alpha=0.25)

        ax.set_title(
            f"Input Component Radar\n{output.category}  |  Score: {output.score:.1f}",
            fontsize=13, fontweight="bold", y=1.12,
        )
        fig.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def dashboard(
        self,
        output: RiskOutput,
        explanation: dict,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        4-panel dashboard:
          [Waterfall] [Radar]
          [Boost breakdown] [Risk narrative text]
        """
        fig = plt.figure(figsize=(18, 12))
        gs  = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

        # ── Panel 1: Component contributions (horizontal bar) ─────────────
        ax1 = fig.add_subplot(gs[0, 0])
        comps = explanation["components"]
        labels = [c["label"] for c in comps]
        vals   = [c["contribution"] for c in comps]
        raws   = [c["raw_score"] for c in comps]
        cols   = [self.COMPONENT_COLOURS[c["component"]] for c in comps]

        bars = ax1.barh(labels, vals, color=cols, alpha=0.85, height=0.55)
        ax1.set_xlabel("Weighted Contribution to Base Score", fontsize=11)
        ax1.set_title("Component Contributions", fontsize=12, fontweight="bold")
        for bar, raw in zip(bars, raws):
            ax1.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                     f"raw: {raw:.0f}", va="center", fontsize=9, color="#555")
        ax1.grid(axis="x", alpha=0.3)
        ax1.set_xlim(0, max(vals) * 1.4 + 5)

        # ── Panel 2: Gauge ────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor("#f8f9fa")
        score  = output.score
        cat    = output.category
        col    = self.CATEGORY_COLOURS.get(cat, "#555")

        # Simple horizontal gauge
        for lo, hi, c, colour, _ in RISK_CATEGORIES:
            ax2.barh(0, hi - lo, left=lo, height=0.6, color=colour, alpha=0.7)
        ax2.barh(0, 2, left=score - 1, height=0.9, color="black", alpha=0.95)

        ax2.set_xlim(0, 100)
        ax2.set_yticks([])
        ax2.set_xticks([0, 30, 60, 80, 100])
        ax2.set_xticklabels(["0\nSAFE", "30", "60\nREVIEW", "80", "100\nBLOCK"], fontsize=9)
        ax2.set_title(
            f"Risk Score: {score:.1f}  │  {cat}",
            fontsize=14, fontweight="bold", color=col,
        )
        ax2.text(score, 0.6, f"▲ {score:.0f}", ha="center", fontsize=13,
                 fontweight="bold", color=col)
        ax2.grid(axis="x", alpha=0.2)

        # ── Panel 3: Boost breakdown ──────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        boost_data  = explanation["boosts"]
        fired_rules = [r for r in boost_data["rules"] if r["fired"]]

        if fired_rules:
            rule_labels = [r["rule"].split(":")[0] for r in fired_rules]
            boost_vals  = [float(r["boost_added"].replace("+", "").replace("×", ""))
                           for r in fired_rules]
            ax3.barh(rule_labels, boost_vals, color=self.BOOST_COLOUR, alpha=0.85, height=0.5)
            ax3.set_xlabel("Boost Amount Added (×)", fontsize=11)
            ax3.set_title(
                f"Boost Engine  (×{boost_data['boost_multiplier']:.3f} total)",
                fontsize=12, fontweight="bold",
            )
            ax3.grid(axis="x", alpha=0.3)
        else:
            ax3.text(0.5, 0.5, "No boosts triggered\n(base score × 1.0)",
                     ha="center", va="center", fontsize=13, color="#95a5a6",
                     transform=ax3.transAxes)
            ax3.set_title("Boost Engine (no boosts)", fontsize=12, fontweight="bold")

        # ── Panel 4: Narrative ────────────────────────────────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_facecolor("#f8f9fa")
        ax4.axis("off")

        narrative = explanation["narrative"]
        # Wrap text
        wrapped = []
        words   = narrative.split()
        line    = ""
        for w in words:
            if len(line) + len(w) + 1 <= 60:
                line += (" " + w if line else w)
            else:
                wrapped.append(line)
                line = w
        if line:
            wrapped.append(line)

        ax4.text(0.05, 0.95, "Risk Narrative", fontsize=12, fontweight="bold",
                 transform=ax4.transAxes, va="top")
        ax4.text(0.05, 0.85, "\n".join(wrapped), fontsize=9, transform=ax4.transAxes,
                 va="top", wrap=True, color="#2c3e50",
                 family="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                           edgecolor="#bdc3c7", alpha=0.8))

        # Counterfactual
        cf = explanation.get("counterfactual", {})
        if "note" in cf and cf["note"]:
            ax4.text(0.05, 0.15, "💡 " + cf["note"],
                     fontsize=8.5, transform=ax4.transAxes, va="bottom",
                     color="#7f8c8d", style="italic",
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffeaa7",
                               edgecolor="#fdcb6e", alpha=0.8))

        fig.suptitle(
            f"Banking Risk Scoring Dashboard  │  Account: {output.account_id or 'N/A'}  "
            f"│  {output.timestamp[:19]}",
            fontsize=14, fontweight="bold", y=1.01,
        )
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — RISK SCORING ENGINE (Main API)
# ════════════════════════════════════════════════════════════════════════════
class BankingRiskScoringEngine:
    """
    Unified banking risk scoring engine.

    Single entry-point for all risk assessment operations.

    Quick Start
    ───────────
    >>> engine = BankingRiskScoringEngine()
    >>> result = engine.score(
    ...     ml_prob        = 0.82,    # raw probability [0, 1] OR [0, 100]
    ...     anomaly_score  = 74.0,    # [0, 100]
    ...     behavior_score = 68.0,    # [0, 100]
    ...     alert_score    = 55.0,    # [0, 100]
    ...     account_id     = "ACC-001",
    ... )
    >>> print(result.summary())
    >>> engine.plot_dashboard(result, save="dashboard.png")

    Batch scoring
    ─────────────
    >>> df_results = engine.score_batch(df)

    Persistence
    ───────────
    >>> engine.save("risk_engine/engine.pkl")
    >>> engine = BankingRiskScoringEngine.load("risk_engine/engine.pkl")
    """

    def __init__(
        self,
        formula:      ScoringFormula  = None,
        explainer:    RiskExplainer   = None,
        visualiser:   RiskVisualiser  = None,
    ):
        self.formula    = formula    or ScoringFormula()
        self.explainer  = explainer  or RiskExplainer()
        self.visualiser = visualiser or RiskVisualiser()

    # ── Single-record scoring ─────────────────────────────────────────────
    def score(
        self,
        ml_prob:        float,
        anomaly_score:  float,
        behavior_score: float,
        alert_score:    float,
        account_id:     str   = "",
        transaction_id: str   = "",
        channel:        str   = "",
        raw_prob:       bool  = True,   # True: ml_prob is [0,1]; False: [0,100]
        explain:        bool  = True,
    ) -> RiskOutput:
        """
        Score a single account/transaction.

        Parameters
        ──────────
        ml_prob        : ML fraud probability. [0,1] if raw_prob=True, else [0,100].
        anomaly_score  : Anomaly detection score [0, 100].
        behavior_score : Transaction behaviour score [0, 100].
        alert_score    : Rule-based alert score [0, 100].
        account_id     : Optional account identifier.
        raw_prob       : If True, ml_prob is treated as [0,1] and multiplied by 100.
        explain        : If True, generate full explanation (slightly slower).
        """
        inp = RiskInput(
            ml_prob        = ml_prob * 100 if raw_prob else ml_prob,
            anomaly_score  = anomaly_score,
            behavior_score = behavior_score,
            alert_score    = alert_score,
            account_id     = account_id,
            transaction_id = transaction_id,
            channel        = channel,
        ).validate()

        output = self.formula.compute(inp)

        if explain:
            output.explanation = self.explainer.explain(output)

        return output

    # ── Batch scoring ─────────────────────────────────────────────────────
    def score_batch(self, df: pd.DataFrame, raw_prob: bool = True) -> pd.DataFrame:
        """
        Score multiple accounts from a DataFrame.

        Expected columns: ml_prob, anomaly_score, behavior_score, alert_score
        Optional:         account_id, transaction_id, channel
        """
        if raw_prob and "ml_prob" in df.columns:
            df = df.copy()
            df["ml_prob"] = df["ml_prob"] * 100
        return self.formula.batch_compute(df)

    # ── Plotting ──────────────────────────────────────────────────────────
    def plot_waterfall(self, output: RiskOutput, save: str = None) -> plt.Figure:
        return self.visualiser.waterfall_chart(output, save_path=save)

    def plot_radar(self, output: RiskOutput, save: str = None) -> plt.Figure:
        return self.visualiser.component_radar(output, save_path=save)

    def plot_dashboard(
        self,
        output: RiskOutput,
        save:   str = None,
    ) -> plt.Figure:
        if not output.explanation:
            output.explanation = self.explainer.explain(output)
        return self.visualiser.dashboard(output, output.explanation, save_path=save)

    # ── Reporting ─────────────────────────────────────────────────────────
    def report(self, output: RiskOutput) -> dict:
        """Full JSON-serialisable report."""
        if not output.explanation:
            output.explanation = self.explainer.explain(output)
        return {
            "risk_score":    output.score,
            "risk_category": output.category,
            "risk_action":   output.category_action,
            "account_id":    output.account_id,
            "timestamp":     output.timestamp,
            "inputs":        output.inputs,
            "components":    output.components,
            "boost": {
                "base_score":       output.boost.base_score,
                "multiplier":       output.boost.boost_multiplier,
                "boost_added":      round(output.boost.total_boost_added(), 4),
                "n_elevated":       output.boost.n_elevated,
                "high_signal_boosts": output.boost.high_signal_boosts,
                "critical_alert":   output.boost.critical_alert_boost,
                "interaction":      output.boost.interaction_boost,
            },
            "explanation": output.explanation,
        }

    def save_report(self, output: RiskOutput, path: str):
        with open(path, "w") as f:
            json.dump(self.report(output), f, indent=2, default=str)
        log.info(f"Report saved → {path}")

    # ── Persistence ───────────────────────────────────────────────────────
    def save(self, path: str = "risk_engine/engine.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        log.info(f"Engine saved → {path}")

    @classmethod
    def load(cls, path: str = "risk_engine/engine.pkl") -> "BankingRiskScoringEngine":
        engine = joblib.load(path)
        log.info(f"Engine loaded ← {path}")
        return engine


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — DEMONSTRATION & TESTING
# ════════════════════════════════════════════════════════════════════════════
def run_demo():
    """
    Demonstrates the risk scoring engine with 6 test scenarios covering
    all four risk categories.
    """
    engine = BankingRiskScoringEngine()

    test_cases = [
        {
            "name":           "High-confidence mule (BLOCK)",
            "ml_prob":        0.95,
            "anomaly_score":  88.0,
            "behavior_score": 91.0,
            "alert_score":    87.0,
            "account_id":     "ACC-001",
        },
        {
            "name":           "ML flagged, low anomaly (REVIEW)",
            "ml_prob":        0.78,
            "anomaly_score":  45.0,
            "behavior_score": 72.0,
            "alert_score":    60.0,
            "account_id":     "ACC-002",
        },
        {
            "name":           "Anomaly signal only (MONITOR)",
            "ml_prob":        0.22,
            "anomaly_score":  68.0,
            "behavior_score": 38.0,
            "alert_score":    42.0,
            "account_id":     "ACC-003",
        },
        {
            "name":           "Alert rule only (MONITOR-borderline)",
            "ml_prob":        0.15,
            "anomaly_score":  30.0,
            "behavior_score": 25.0,
            "alert_score":    88.0,
            "account_id":     "ACC-004",
        },
        {
            "name":           "Low risk — legitimate (SAFE)",
            "ml_prob":        0.03,
            "anomaly_score":  12.0,
            "behavior_score": 18.0,
            "alert_score":    8.0,
            "account_id":     "ACC-005",
        },
        {
            "name":           "Borderline case (SAFE/MONITOR boundary)",
            "ml_prob":        0.28,
            "anomaly_score":  35.0,
            "behavior_score": 30.0,
            "alert_score":    22.0,
            "account_id":     "ACC-006",
        },
    ]

    print("\n" + "═"*70)
    print("  BANKING RISK SCORING ENGINE — DEMONSTRATION")
    print("═"*70)
    print(f"  {'Account':<12} {'Test Case':<38} {'Score':>7} {'Category':<10}")
    print("  " + "─"*65)

    all_outputs = []
    for tc in test_cases:
        out = engine.score(
            ml_prob        = tc["ml_prob"],
            anomaly_score  = tc["anomaly_score"],
            behavior_score = tc["behavior_score"],
            alert_score    = tc["alert_score"],
            account_id     = tc["account_id"],
            explain        = True,
        )
        all_outputs.append((tc, out))
        cat_indicator = {
            "BLOCK": "🔴", "REVIEW": "🟠", "MONITOR": "🟡", "SAFE": "🟢"
        }.get(out.category, "⚪")
        print(f"  {tc['account_id']:<12} {tc['name'][:37]:<38} "
              f"{out.score:>6.1f}  {cat_indicator} {out.category}")

    print("═"*70)

    # Detailed report for the highest-risk case
    tc_hi, out_hi = all_outputs[0]
    print(f"\n{'─'*60}")
    print(f"  DETAILED REPORT — {tc_hi['account_id']} ({tc_hi['name']})")
    print("─"*60)
    print(out_hi.summary())

    # Narrative
    print(f"\n  NARRATIVE EXPLANATION:")
    print("  " + out_hi.explanation["narrative"])

    # Boost breakdown
    boosts = out_hi.explanation["boosts"]
    print(f"\n  BOOST BREAKDOWN (×{boosts['boost_multiplier']:.3f} total):")
    for rule in boosts["rules"]:
        fired = "✓" if rule["fired"] else "✗"
        print(f"    [{fired}] {rule['rule']}  →  {rule['boost_added']}  "
              f"(value: {rule['value']})")

    # Counterfactual
    cf = out_hi.explanation["counterfactual"]
    print(f"\n  COUNTERFACTUAL (how to lower the risk):")
    if cf.get("note"):
        print(f"    {cf['note']}")

    # Generate plots for first case
    print(f"\n  Generating plots …")
    engine.plot_dashboard(
        out_hi,
        save=str(OUTPUT_DIR / "plots" / f"dashboard_{out_hi.account_id}.png"),
    )
    plt.close("all")

    engine.plot_waterfall(
        out_hi,
        save=str(OUTPUT_DIR / "plots" / f"waterfall_{out_hi.account_id}.png"),
    )
    plt.close("all")

    engine.plot_radar(
        out_hi,
        save=str(OUTPUT_DIR / "plots" / f"radar_{out_hi.account_id}.png"),
    )
    plt.close("all")

    # Save JSON report
    engine.save_report(out_hi, str(OUTPUT_DIR / f"report_{out_hi.account_id}.json"))

    # Batch scoring demo
    print(f"\n  Batch scoring demo …")
    df_batch = pd.DataFrame([
        {
            "account_id": tc["account_id"],
            "ml_prob":        tc["ml_prob"],
            "anomaly_score":  tc["anomaly_score"],
            "behavior_score": tc["behavior_score"],
            "alert_score":    tc["alert_score"],
        }
        for tc, _ in all_outputs
    ])
    df_results = engine.score_batch(df_batch, raw_prob=True)
    print("\n" + df_results[["account_id", "risk_score", "risk_category",
                              "boost_multiplier", "n_elevated"]].to_string(index=False))

    # Save batch results
    df_results.to_csv(OUTPUT_DIR / "batch_results.csv", index=False)
    print(f"\n  Batch results → {OUTPUT_DIR}/batch_results.csv")

    # Save engine
    engine.save(str(OUTPUT_DIR / "engine.pkl"))

    print(f"\n✅  Demo complete. Outputs in '{OUTPUT_DIR}/'")
    return engine, all_outputs


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — INTEGRATION HELPER (with existing pipeline)
# ════════════════════════════════════════════════════════════════════════════
def integrate_with_fraud_pipeline(
    engine: BankingRiskScoringEngine,
    df_accounts: pd.DataFrame,
    clf_pipeline_path: str = "models/inference_pipeline.pkl",
    anomaly_pipeline_path: str = "anomaly_detection/models/combined_pipeline.pkl",
    behavior_col: str = None,
    alert_col:    str = None,
) -> pd.DataFrame:
    """
    End-to-end integration: loads the fraud classifier + anomaly detector,
    computes all scores, and passes them through the risk scoring engine.

    If behavior_col / alert_col are not in the DataFrame, they default to 0.

    Parameters
    ──────────
    df_accounts           : Raw account feature DataFrame
    clf_pipeline_path     : Path to MuleAccountInferencePipeline pickle
    anomaly_pipeline_path : Path to CombinedFraudPipeline pickle (optional)
    behavior_col          : Column name in df containing pre-computed behaviour score
    alert_col             : Column name in df containing pre-computed alert score

    Returns
    ───────
    DataFrame with all raw scores + final risk assessment
    """
    import sys
    sys.path.insert(0, ".")

    scores = pd.DataFrame(index=df_accounts.index)

    # ── ML probability ────────────────────────────────────────────────────
    try:
        from fraud_detection_model import MuleAccountInferencePipeline
        clf    = MuleAccountInferencePipeline.load(clf_pipeline_path)
        clf_r  = clf.predict(df_accounts)
        scores["ml_prob"] = clf_r["mule_prob"].values * 100
        log.info("  ✓ ML fraud probabilities loaded from inference_pipeline")
    except Exception as e:
        log.warning(f"  ✗ Could not load classifier: {e}. Using ml_prob=0.")
        scores["ml_prob"] = 0.0

    # ── Anomaly score ─────────────────────────────────────────────────────
    try:
        from anomaly_layer import AnomalyDetectionPipeline
        ad = AnomalyDetectionPipeline.load()
        scores["anomaly_score"] = ad.score(df_accounts)
        log.info("  ✓ Anomaly scores loaded from anomaly pipeline")
    except Exception as e:
        log.warning(f"  ✗ Could not load anomaly pipeline: {e}. Using anomaly_score=0.")
        scores["anomaly_score"] = 0.0

    # ── Behaviour / Alert scores ──────────────────────────────────────────
    scores["behavior_score"] = (
        df_accounts[behavior_col].values if behavior_col and behavior_col in df_accounts
        else 0.0
    )
    scores["alert_score"] = (
        df_accounts[alert_col].values if alert_col and alert_col in df_accounts
        else 0.0
    )

    # ── Risk scoring ──────────────────────────────────────────────────────
    log.info("  Computing risk scores …")
    risk_df = engine.score_batch(scores, raw_prob=False)  # already in [0,100]
    return pd.concat([df_accounts.reset_index(drop=True), risk_df], axis=1)


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_demo()
