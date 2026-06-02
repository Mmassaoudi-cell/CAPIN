"""
capinn.py — Constraint-Aware Physics-Informed Neural Network (CAPIN)
for Intrusion Detection in Cyber-Physical Smart Grids

Reference:
    "A Constraint-Aware Physics-Informed Neural Network for Enhanced
     Intrusion Detection in Cyber-Physical Smart Grids"
    Mohamed Massaoudi, Maymouna Ez Eddin

Dataset:
    Mississippi State University ICS Cyberattack Dataset (binary split).
    https://www.ece.msstate.edu/~pvs/files/research/ICS_Cyberattack_Dataset.html

Usage:
    python capinn.py --data_dir /path/to/binaryAllNaturalPlusNormalVsAttacks
    python capinn.py --data_dir ./data --seed 2026 --no_nn
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

# Suppress multi-threading warnings from sklearn / numpy backends
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    XGBClassifier = None
    HAS_XGB = False

EPS = 1e-9
SEEDS = [2026, 2027, 2028]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PreparedData:
    """Holds all data partitions produced by prepare_data()."""
    x_raw: pd.DataFrame
    x_physics: pd.DataFrame
    constraints: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray
    feature_names: list[str]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(data_dir: Path) -> pd.DataFrame:
    """
    Read all data*.csv files in data_dir, tag each with a scenario-block
    index, and return the combined frame filtered to Attack / Natural rows.
    """
    frames = []
    for idx, path in enumerate(sorted(data_dir.glob("data*.csv"))):
        frame = pd.read_csv(path)
        frame["source_file"] = path.stem
        frame["scenario_block"] = idx
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No data*.csv files found in {data_dir}")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["marker"].isin(["Attack", "Natural"])].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Physics-informed feature engineering
# ---------------------------------------------------------------------------

def _cols(columns: list[str], pattern: str) -> list[str]:
    return [c for c in columns if pattern in c]


def build_physics_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct physics-informed features and physical constraint residuals.

    Features (appended to raw measurements):
      - Log / relay: log_sum, log_density, relay_sequence
      - Frequency: freq_mean, freq_std, freq_range, freq_nominal_error,
                   df_abs_mean, df_abs_max
      - Voltage / current: voltage_mean, voltage_std, voltage_imbalance,
                           current_mean, current_std, current_imbalance
      - Per-relay (R1–R4): voltage/current mean & std, pairwise power P_j,
                           relay power mean/std, Ohm-residual
      - Global: power_balance_error, log_voltage_correlation,
                log_current_correlation

    Constraints (used for sample weighting):
      protection_activity, frequency_stability, frequency_derivative,
      voltage_balance, current_balance, power_balance,
      ohm_law_consistency, kirchhoff_current
    """
    columns = raw.columns.tolist()
    physics = pd.DataFrame(index=raw.index)
    constraints = pd.DataFrame(index=raw.index)

    # --- Log / protection features ---
    log_cols = [c for c in columns if "log" in c.lower() or c.endswith(":S")]
    if log_cols:
        logs = raw[log_cols].fillna(0.0)
        physics["log_sum"] = logs.sum(axis=1)
        physics["log_density"] = logs.sum(axis=1) / (logs.gt(0).sum(axis=1) + EPS)
        physics["relay_sequence"] = sum(
            (2 ** i) * logs[c] for i, c in enumerate(log_cols[:12])
        )
        constraints["protection_activity"] = (logs.gt(0).sum(axis=1) > 0).astype(float)

    # --- Frequency features ---
    freq_cols = [c for c in columns if c.endswith(":F")]
    if freq_cols:
        freqs = raw[freq_cols]
        physics["freq_mean"] = freqs.mean(axis=1)
        physics["freq_std"] = freqs.std(axis=1)
        physics["freq_range"] = freqs.max(axis=1) - freqs.min(axis=1)
        physics["freq_nominal_error"] = (freqs.mean(axis=1) - 60.0).abs()
        constraints["frequency_stability"] = physics["freq_range"].abs()

    df_cols = [c for c in columns if c.endswith(":DF")]
    if df_cols:
        dfs = raw[df_cols]
        physics["df_abs_mean"] = dfs.abs().mean(axis=1)
        physics["df_abs_max"] = dfs.abs().max(axis=1)
        constraints["frequency_derivative"] = physics["df_abs_max"]

    # --- Global voltage / current ---
    all_v_cols = _cols(columns, ":V")
    all_i_cols = _cols(columns, ":I")
    if all_v_cols:
        v = raw[all_v_cols]
        physics["voltage_mean"] = v.mean(axis=1)
        physics["voltage_std"] = v.std(axis=1)
        physics["voltage_imbalance"] = v.std(axis=1) / (v.abs().mean(axis=1) + EPS)
        constraints["voltage_balance"] = physics["voltage_imbalance"]
    if all_i_cols:
        i = raw[all_i_cols]
        physics["current_mean"] = i.mean(axis=1)
        physics["current_std"] = i.std(axis=1)
        physics["current_imbalance"] = i.std(axis=1) / (i.abs().mean(axis=1) + EPS)
        constraints["current_balance"] = physics["current_imbalance"]

    # --- Per-relay features (R1–R4) ---
    relay_power_means: list[pd.Series] = []
    relay_ohm_errors: list[pd.Series] = []
    relay_kcl_errors: list[pd.Series] = []

    for relay in range(1, 5):
        v_cols = [f"R{relay}-PM{k}:V" for k in range(1, 13)
                  if f"R{relay}-PM{k}:V" in columns]
        i_cols = [f"R{relay}-PM{k}:I" for k in range(4, 13)
                  if f"R{relay}-PM{k}:I" in columns]
        z_col = f"R{relay}-PA:Z"

        if v_cols:
            v = raw[v_cols]
            physics[f"R{relay}_voltage_mean"] = v.mean(axis=1)
            physics[f"R{relay}_voltage_std"] = v.std(axis=1)
        if i_cols:
            cur = raw[i_cols]
            physics[f"R{relay}_current_mean"] = cur.mean(axis=1)
            physics[f"R{relay}_current_std"] = cur.std(axis=1)
            # Kirchhoff current law proxy: |sum(I)| / sum(|I|)
            relay_kcl_errors.append(
                cur.sum(axis=1).abs() / (cur.abs().sum(axis=1) + EPS)
            )
        if v_cols and i_cols:
            pairs = min(len(v_cols), len(i_cols))
            p_terms: list[pd.Series] = []
            ohm_terms: list[pd.Series] = []
            for j in range(pairs):
                p = raw[v_cols[j]] * raw[i_cols[j]]
                physics[f"R{relay}_P{j + 1}"] = p
                p_terms.append(p)
                if z_col in columns:
                    z = raw[z_col].abs()
                    ohm_terms.append(
                        (raw[v_cols[j]].abs() - raw[i_cols[j]].abs() * z).abs()
                        / (raw[v_cols[j]].abs() + EPS)
                    )
            power = pd.concat(p_terms, axis=1)
            physics[f"R{relay}_power_mean"] = power.mean(axis=1)
            physics[f"R{relay}_power_std"] = power.std(axis=1)
            relay_power_means.append(power.mean(axis=1))
            if ohm_terms:
                ohm_error = pd.concat(ohm_terms, axis=1).mean(axis=1)
                physics[f"R{relay}_ohm_error"] = ohm_error
                relay_ohm_errors.append(ohm_error)

    # --- Inter-relay aggregates ---
    if relay_power_means:
        powers = pd.concat(relay_power_means, axis=1)
        physics["power_balance_error"] = powers.std(axis=1) / (
            powers.abs().mean(axis=1) + EPS
        )
        constraints["power_balance"] = physics["power_balance_error"]
    if relay_ohm_errors:
        constraints["ohm_law_consistency"] = pd.concat(
            relay_ohm_errors, axis=1
        ).mean(axis=1)
    if relay_kcl_errors:
        constraints["kirchhoff_current"] = pd.concat(
            relay_kcl_errors, axis=1
        ).mean(axis=1)

    # --- Cross-domain: log × electrical ---
    if all_v_cols and log_cols:
        centered_v = (
            raw[all_v_cols].mean(axis=1) - raw[all_v_cols].mean().mean()
        )
        physics["log_voltage_correlation"] = (
            physics.get("log_sum", 0.0) * centered_v
        )
    if all_i_cols and log_cols:
        centered_i = (
            raw[all_i_cols].mean(axis=1) - raw[all_i_cols].mean().mean()
        )
        physics["log_current_correlation"] = (
            physics.get("log_sum", 0.0) * centered_i
        )

    physics = physics.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    constraints = constraints.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return physics, constraints


def normalize_frame(
    frame: pd.DataFrame, fit_index: np.ndarray | None = None
) -> pd.DataFrame:
    """Clip to training [1st, 99th] percentile range then scale to [0, 1]."""
    result = pd.DataFrame(index=frame.index)
    train = frame.iloc[fit_index] if fit_index is not None else frame
    for col in frame.columns:
        lo = float(train[col].quantile(0.01))
        hi = float(train[col].quantile(0.99))
        denom = hi - lo
        if abs(denom) < EPS:
            result[col] = 0.0
        else:
            result[col] = ((frame[col].clip(lo, hi) - lo) / denom).clip(0.0, 1.0)
    return result


# ---------------------------------------------------------------------------
# Full data preparation pipeline
# ---------------------------------------------------------------------------

def prepare_data(data_dir: Path) -> PreparedData:
    """Load CSVs, impute, engineer features, and return PreparedData."""
    df = load_dataset(data_dir)
    y = (df["marker"] == "Attack").astype(int).to_numpy()
    groups = df["scenario_block"].to_numpy()

    drop_cols = {"marker", "source_file", "scenario_block"}
    x_raw = df[[c for c in df.columns if c not in drop_cols]].apply(
        pd.to_numeric, errors="coerce"
    )
    imputer = SimpleImputer(strategy="median")
    x_raw = pd.DataFrame(
        imputer.fit_transform(x_raw),
        columns=x_raw.columns,
        index=x_raw.index,
    )
    x_physics, constraints = build_physics_features(x_raw)
    return PreparedData(x_raw, x_physics, constraints, y, groups, x_raw.columns.tolist())


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _make_xgb(seed: int, scale_pos_weight: float):
    if not HAS_XGB:
        return RandomForestClassifier(
            n_estimators=50, n_jobs=1, class_weight="balanced", random_state=seed
        )
    return XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.85,
        min_child_weight=2.0,
        reg_lambda=1.5,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=1,
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
    )


def _fit_predict_proba(
    model, x_train, y_train, x_eval, sample_weight=None
) -> np.ndarray:
    try:
        if sample_weight is None:
            model.fit(x_train, y_train)
        else:
            model.fit(x_train, y_train, sample_weight=sample_weight)
    except (TypeError, ValueError):
        model.fit(x_train, y_train)
    return _predict_proba(model, x_eval)


def _predict_proba(model, x_eval) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_eval)[:, 1]
    score = model.decision_function(x_eval)
    return 1.0 / (1.0 + np.exp(-score))


def optimize_weights(preds: list[np.ndarray], y_val: np.ndarray) -> np.ndarray:
    """SLSQP-optimized ensemble weights minimising validation log-loss."""
    mat = np.vstack(preds).T
    n_models = mat.shape[1]

    def loss(w: np.ndarray) -> float:
        p = np.clip(mat @ w, EPS, 1.0 - EPS)
        return -np.mean(y_val * np.log(p) + (1 - y_val) * np.log(1 - p))

    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * n_models
    result = minimize(
        loss, np.ones(n_models) / n_models,
        bounds=bounds, constraints=cons, method="SLSQP",
    )
    return result.x if result.success else np.ones(n_models) / n_models


def optimize_threshold(y_val: np.ndarray, prob_val: np.ndarray) -> float:
    """
    Select decision threshold on the validation set by maximising a combined
    score: 0.5 * F1 + 0.5 * balanced_accuracy - 0.15 * FPR.
    This penalises false alarms while preserving recall for attack detection.
    """
    candidates = np.linspace(0.20, 0.80, 121)
    scores = []
    for t in candidates:
        pred = (prob_val >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn + EPS)
        tnr = tn / (tn + fp + EPS)
        fpr = fp / (fp + tn + EPS)
        balanced_acc = 0.5 * (tpr + tnr)
        scores.append(0.5 * f1_score(y_val, pred) + 0.5 * balanced_acc - 0.15 * fpr)
    return float(candidates[int(np.argmax(scores))])


def compute_metrics(
    name: str, y_true: np.ndarray, prob: np.ndarray,
    threshold: float, train_time: float = 0.0,
) -> dict:
    """Return a dict of all reported metrics for one model evaluation."""
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    two_classes = len(np.unique(y_true)) == 2
    return {
        "model": name,
        "samples": int(len(y_true)),
        "normal": int((y_true == 0).sum()),
        "attack": int((y_true == 1).sum()),
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, prob), 4) if two_classes else float("nan"),
        "pr_auc": round(average_precision_score(y_true, prob), 4) if two_classes else float("nan"),
        "fpr": round(fp / (fp + tn + EPS), 4),
        "fnr": round(fn / (fn + tp + EPS), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": round(threshold, 4),
        "train_time_sec": round(train_time, 2),
    }


# ---------------------------------------------------------------------------
# CAPIN model
# ---------------------------------------------------------------------------

class CapinModel:
    """
    Constraint-Aware Physics-Informed Neural Network.

    Components:
      - Physics-informed feature engineering (build_physics_features)
      - Constraint-guided sample weighting: ω_i = ω_class * (1 + 0.35 * C̄)
      - Ensemble: XGBoost + Random Forest + Extra Trees + MLP
      - Validation-optimised ensemble weights (SLSQP)
      - Validation-optimised decision threshold

    Parameters
    ----------
    seed : int
        Global random seed for reproducibility.
    use_physics : bool
        Include physics-engineered features in the input matrix.
    use_constraints : bool
        Apply constraint-score sample weighting during training.
    use_nn : bool
        Include the MLP branch in the ensemble.
    use_ensemble : bool
        Include tree-based models (XGBoost, RF, Extra Trees) in the ensemble.
    """

    def __init__(
        self,
        seed: int = 2026,
        use_physics: bool = True,
        use_constraints: bool = True,
        use_nn: bool = True,
        use_ensemble: bool = True,
    ) -> None:
        self.seed = seed
        self.use_physics = use_physics
        self.use_constraints = use_constraints
        self.use_nn = use_nn
        self.use_ensemble = use_ensemble
        self.weights: np.ndarray | None = None
        self.threshold: float = 0.5
        self.models: list = []

    def _build_feature_matrix(
        self, prepared: PreparedData, indices: np.ndarray
    ) -> pd.DataFrame:
        parts = [prepared.x_raw]
        if self.use_physics:
            parts.append(prepared.x_physics)
        if self.use_constraints:
            c_features = normalize_frame(prepared.constraints).add_prefix("constraint_")
            parts.append(c_features)
        return pd.concat(parts, axis=1).iloc[indices]

    def _compute_sample_weights(
        self,
        prepared: PreparedData,
        train_idx: np.ndarray,
        y_train: np.ndarray,
    ) -> np.ndarray:
        neg = max((y_train == 0).sum(), 1)
        pos = max((y_train == 1).sum(), 1)
        class_w = np.where(y_train == 1, neg / pos, 1.0)
        if not self.use_constraints:
            return class_w
        c_norm = normalize_frame(prepared.constraints, fit_index=train_idx).iloc[train_idx]
        if c_norm.empty:
            return class_w
        c_score = c_norm.mean(axis=1).to_numpy()
        # Physics-guided weighting: λ_c = 0.35 (selected on validation set)
        return class_w * (1.0 + 0.35 * c_score)

    def fit(
        self,
        prepared: PreparedData,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
    ) -> None:
        x_train = self._build_feature_matrix(prepared, train_idx)
        x_val = self._build_feature_matrix(prepared, val_idx)
        y_train = prepared.y[train_idx]
        y_val = prepared.y[val_idx]
        sample_weight = self._compute_sample_weights(prepared, train_idx, y_train)
        scale_pos = max((y_train == 0).sum(), 1) / max((y_train == 1).sum(), 1)

        self.models = []
        val_probs: list[np.ndarray] = []

        if self.use_ensemble:
            tree_models = [
                _make_xgb(self.seed, scale_pos),
                RandomForestClassifier(
                    n_estimators=50, max_depth=None, min_samples_leaf=2,
                    n_jobs=1, class_weight="balanced_subsample", random_state=self.seed,
                ),
                ExtraTreesClassifier(
                    n_estimators=50, max_depth=None, min_samples_leaf=1,
                    n_jobs=1, class_weight="balanced", random_state=self.seed,
                ),
            ]
            for model in tree_models:
                val_probs.append(
                    _fit_predict_proba(model, x_train, y_train, x_val, sample_weight)
                )
                self.models.append(model)

        if self.use_nn:
            nn = make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    activation="relu",
                    alpha=1e-4,
                    learning_rate_init=5e-4,
                    max_iter=15,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=8,
                    random_state=self.seed,
                ),
            )
            val_probs.append(_fit_predict_proba(nn, x_train, y_train, x_val))
            self.models.append(nn)

        self.weights = optimize_weights(val_probs, y_val)
        p_val = np.vstack(val_probs).T @ self.weights
        self.threshold = optimize_threshold(y_val, p_val)

    def predict_proba(self, prepared: PreparedData, indices: np.ndarray) -> np.ndarray:
        x_eval = self._build_feature_matrix(prepared, indices)
        probs = [_predict_proba(model, x_eval) for model in self.models]
        return np.vstack(probs).T @ self.weights

    def predict(self, prepared: PreparedData, indices: np.ndarray) -> np.ndarray:
        return (self.predict_proba(prepared, indices) >= self.threshold).astype(int)


# ---------------------------------------------------------------------------
# Train / validation / test splits
# ---------------------------------------------------------------------------

def stratified_split(
    y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """64 / 16 / 20 stratified split (train / val / test)."""
    idx = np.arange(len(y))
    train_val, test = train_test_split(idx, test_size=0.20, stratify=y, random_state=seed)
    train, val = train_test_split(
        train_val, test_size=0.20, stratify=y[train_val], random_state=seed
    )
    return train, val, test


def scenario_block_split(
    y: np.ndarray, groups: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """80 / 20 group (scenario-block) holdout split."""
    idx = np.arange(len(y))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_val_pos, test_pos = next(gss.split(idx, y, groups))
    train_val = idx[train_val_pos]
    test = idx[test_pos]
    train, val = train_test_split(
        train_val, test_size=0.20, stratify=y[train_val], random_state=seed
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_baselines(
    prepared: PreparedData,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    seed: int,
) -> list[dict]:
    """Train and evaluate five conventional intrusion-detection baselines."""
    x_train = prepared.x_raw.iloc[train]
    x_val   = prepared.x_raw.iloc[val]
    x_test  = prepared.x_raw.iloc[test]
    y_train, y_val, y_test = prepared.y[train], prepared.y[val], prepared.y[test]
    scale_pos = max((y_train == 0).sum(), 1) / max((y_train == 1).sum(), 1)

    baselines = [
        ("ANN", make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32, 16), max_iter=15,
                early_stopping=True, random_state=seed,
            ),
        )),
        ("Logistic Regression", make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced", C=0.5, max_iter=1000, random_state=seed,
            ),
        )),
        ("Random Forest", RandomForestClassifier(
            n_estimators=50, n_jobs=1, class_weight="balanced_subsample",
            random_state=seed,
        )),
        ("AdaBoost", AdaBoostClassifier(
            n_estimators=50, learning_rate=0.55, random_state=seed,
        )),
        ("XGBoost", _make_xgb(seed, scale_pos)),
    ]

    rows = []
    for name, model in baselines:
        start = time.time()
        prob_val = _fit_predict_proba(model, x_train, y_train, x_val)
        threshold = optimize_threshold(y_val, prob_val)
        prob_test = _predict_proba(model, x_test)
        rows.append(compute_metrics(name, y_test, prob_test, threshold, time.time() - start))
    return rows


def run_ablation(
    prepared: PreparedData,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    seed: int,
) -> list[dict]:
    """Ablation study: evaluate each CAPIN component in isolation."""
    variants = [
        ("Full CAPIN",
         dict(use_physics=True,  use_constraints=True,  use_nn=True,  use_ensemble=True)),
        ("No Physics Features",
         dict(use_physics=False, use_constraints=True,  use_nn=True,  use_ensemble=True)),
        ("No Constraint Weights",
         dict(use_physics=True,  use_constraints=False, use_nn=True,  use_ensemble=True)),
        ("NN Branch Only",
         dict(use_physics=True,  use_constraints=False, use_nn=True,  use_ensemble=False)),
        ("Tree Ensemble Only",
         dict(use_physics=True,  use_constraints=True,  use_nn=False, use_ensemble=True)),
    ]
    rows = []
    for name, kwargs in variants:
        model = CapinModel(seed=seed, **kwargs)
        start = time.time()
        model.fit(prepared, train, val)
        prob = model.predict_proba(prepared, test)
        rows.append(compute_metrics(name, prepared.y[test], prob, model.threshold,
                                    time.time() - start))
    return rows


def run_data_dependency(
    prepared: PreparedData,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    seed: int,
) -> list[dict]:
    """Evaluate CAPIN performance as the training-set fraction varies."""
    rows = []
    for frac in [0.10, 0.25, 0.50, 0.75, 1.00]:
        if frac >= 0.999:
            subtrain = train
        else:
            size = max(1000, int(round(frac * len(train))))
            rng = np.random.default_rng(seed + int(frac * 100))
            subtrain = rng.choice(train, size=size, replace=False)

        model = CapinModel(
            seed=seed + int(frac * 100),
            use_physics=True, use_constraints=True,
            use_nn=False, use_ensemble=True,
        )
        start = time.time()
        model.fit(prepared, subtrain, val)
        prob = model.predict_proba(prepared, test)
        row = compute_metrics("CAPIN", prepared.y[test], prob, model.threshold,
                              time.time() - start)
        row["training_fraction"] = round(frac, 2)
        row["training_samples"] = int(len(subtrain))
        rows.append(row)
    return rows


def run_attack_intensity(
    prepared: PreparedData,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    seed: int,
) -> list[dict]:
    """
    Evaluate CAPIN across attack prevalence levels (5 %, 10 %, 20 %, 30 %).
    The full normal test partition is retained for every run; attack samples
    are subsampled to hit the target prevalence. Each level is repeated with
    three seeds to report mean ± std.
    """
    model = CapinModel(seed=seed, use_physics=True, use_constraints=True,
                       use_nn=True, use_ensemble=True)
    model.fit(prepared, train, val)

    y_test = prepared.y[test]
    normal_idx = test[y_test == 0]
    attack_idx = test[y_test == 1]

    rows = []
    for alpha in [0.05, 0.10, 0.20, 0.30]:
        attack_needed = min(
            int(round(alpha / (1 - alpha) * len(normal_idx))),
            len(attack_idx),
        )
        for rep, rep_seed in enumerate(SEEDS):
            rng = np.random.default_rng(rep_seed + int(alpha * 1000))
            chosen_attack = rng.choice(attack_idx, size=attack_needed, replace=False)
            eval_idx = np.concatenate([normal_idx, chosen_attack])
            prob = model.predict_proba(prepared, eval_idx)
            row = compute_metrics("CAPIN", prepared.y[eval_idx], prob, model.threshold)
            row["attack_intensity"] = alpha
            row["repeat_seed"] = rep_seed
            rows.append(row)
    return rows


def summarize_intensity(rows: list[dict]) -> list[dict]:
    """Aggregate per-intensity repeated measurements into mean ± std rows."""
    from collections import defaultdict
    grouped: dict[float, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[r["attack_intensity"]].append(r)

    summary = []
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fpr", "fnr"]
    for alpha in sorted(grouped):
        group = grouped[alpha]
        row: dict = {"attack_intensity": alpha}
        for m in metrics:
            vals = [g[m] for g in group]
            row[f"{m}_mean"] = round(float(np.mean(vals)), 4)
            row[f"{m}_std"] = round(float(np.std(vals, ddof=0)), 4)
        for col in ["samples", "normal", "attack", "tn", "fp", "fn", "tp"]:
            row[col] = int(round(np.mean([g[col] for g in group])))
        summary.append(row)
    return summary


# ---------------------------------------------------------------------------
# Console reporting (no tables or graphs)
# ---------------------------------------------------------------------------

def _print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def _print_metrics(row: dict) -> None:
    print(
        f"  {row['model']:<30}"
        f"  Acc={row['accuracy']:.4f}"
        f"  Prec={row['precision']:.4f}"
        f"  Rec={row['recall']:.4f}"
        f"  F1={row['f1']:.4f}"
        f"  ROC={row['roc_auc']:.4f}"
        f"  PR={row['pr_auc']:.4f}"
        f"  FPR={row['fpr']:.4f}"
        f"  FNR={row['fnr']:.4f}"
        f"  FP={row['fp']}  FN={row['fn']}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CAPIN: Constraint-Aware Physics-Informed Neural Network "
                    "for Smart Grid Intrusion Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Folder containing the binary ICS dataset CSV files (data1.csv … data15.csv).",
    )
    parser.add_argument(
        "--seed", type=int, default=2026, help="Master random seed."
    )
    parser.add_argument(
        "--out_dir", type=Path, default=None,
        help="Optional directory to save CSV result files.",
    )
    parser.add_argument(
        "--no_nn", action="store_true",
        help="Disable the MLP branch (faster, useful for quick experiments).",
    )
    parser.add_argument(
        "--skip_baselines", action="store_true",
        help="Skip baseline model evaluation.",
    )
    parser.add_argument(
        "--skip_ablation", action="store_true",
        help="Skip the ablation study.",
    )
    parser.add_argument(
        "--skip_intensity", action="store_true",
        help="Skip the attack-intensity analysis.",
    )
    parser.add_argument(
        "--skip_data_dependency", action="store_true",
        help="Skip the data-dependency experiment.",
    )
    parser.add_argument(
        "--skip_scenario_block", action="store_true",
        help="Skip the scenario-block holdout validation.",
    )
    args = parser.parse_args()

    # --- Load and prepare data ---
    print(f"\nLoading dataset from: {args.data_dir}")
    t0 = time.time()
    prepared = prepare_data(args.data_dir)
    print(
        f"  {len(prepared.y):,} samples  |  "
        f"normal={int((prepared.y == 0).sum()):,}  attack={int((prepared.y == 1).sum()):,}  |  "
        f"raw features={prepared.x_raw.shape[1]}  "
        f"physics features={prepared.x_physics.shape[1]}  "
        f"constraints={prepared.constraints.shape[1]}"
    )
    print(f"  Load time: {time.time() - t0:.1f}s")

    # --- Splits ---
    train, val, test = stratified_split(prepared.y, args.seed)
    g_train, g_val, g_test = scenario_block_split(prepared.y, prepared.groups, args.seed)

    print(
        f"\nStratified split  — train={len(train):,}  val={len(val):,}  test={len(test):,}"
    )
    print(
        f"Scenario-block split — train={len(g_train):,}  val={len(g_val):,}  test={len(g_test):,}"
    )

    results: dict[str, list[dict]] = {}

    # --- Main CAPIN evaluation ---
    _print_section("CAPIN — Main Evaluation (stratified split)")
    capin = CapinModel(seed=args.seed, use_nn=not args.no_nn)
    t0 = time.time()
    capin.fit(prepared, train, val)
    train_time = time.time() - t0
    prob_test = capin.predict_proba(prepared, test)
    row = compute_metrics("Full CAPIN", prepared.y[test], prob_test, capin.threshold, train_time)
    _print_metrics(row)
    print(f"  Ensemble weights: {[round(w, 4) for w in capin.weights.tolist()]}")
    print(f"  Decision threshold: {capin.threshold:.4f}")
    print(f"  Training time: {train_time:.1f}s")
    results["capin_main"] = [row]

    # --- Baselines ---
    if not args.skip_baselines:
        _print_section("Baseline Models vs CAPIN")
        baseline_rows = run_baselines(prepared, train, val, test, args.seed)
        for r in baseline_rows:
            _print_metrics(r)
        _print_metrics(row)
        results["baselines"] = baseline_rows

    # --- Ablation ---
    if not args.skip_ablation:
        _print_section("Ablation Study")
        ablation_rows = run_ablation(prepared, train, val, test, args.seed)
        for r in ablation_rows:
            _print_metrics(r)
        results["ablation"] = ablation_rows

    # --- Data dependency ---
    if not args.skip_data_dependency:
        _print_section("Data-Dependency Analysis (training fraction)")
        dep_rows = run_data_dependency(prepared, train, val, test, args.seed)
        for r in dep_rows:
            print(
                f"  fraction={r['training_fraction']:.2f}"
                f"  train_n={r['training_samples']:>6,}"
                f"  Acc={r['accuracy']:.4f}  F1={r['f1']:.4f}"
                f"  ROC={r['roc_auc']:.4f}"
            )
        results["data_dependency"] = dep_rows

    # --- Attack intensity ---
    if not args.skip_intensity:
        _print_section("Attack-Intensity Analysis (repeated subsampling)")
        intensity_rows = run_attack_intensity(prepared, train, val, test, args.seed)
        summary = summarize_intensity(intensity_rows)
        for s in summary:
            print(
                f"  α={s['attack_intensity']:.2f}"
                f"  Acc={s['accuracy_mean']:.4f}±{s['accuracy_std']:.4f}"
                f"  Rec={s['recall_mean']:.4f}±{s['recall_std']:.4f}"
                f"  F1={s['f1_mean']:.4f}±{s['f1_std']:.4f}"
                f"  ROC={s['roc_auc_mean']:.4f}±{s['roc_auc_std']:.4f}"
                f"  PR={s['pr_auc_mean']:.4f}±{s['pr_auc_std']:.4f}"
                f"  FPR={s['fpr_mean']:.4f}±{s['fpr_std']:.4f}"
            )
        results["attack_intensity"] = intensity_rows
        results["attack_intensity_summary"] = summary

    # --- Scenario-block holdout ---
    if not args.skip_scenario_block:
        _print_section("Scenario-Block Holdout Validation")
        scenario_capin = CapinModel(seed=args.seed, use_nn=not args.no_nn)
        t0 = time.time()
        scenario_capin.fit(prepared, g_train, g_val)
        scenario_prob = scenario_capin.predict_proba(prepared, g_test)
        s_row = compute_metrics(
            "CAPIN (scenario-block)",
            prepared.y[g_test], scenario_prob,
            scenario_capin.threshold, time.time() - t0,
        )
        _print_metrics(s_row)
        print("  (Expected lower scores due to complete acquisition-block holdout)")
        results["scenario_block"] = [s_row]

    # --- Optional CSV export ---
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for key, rows_list in results.items():
            path = args.out_dir / f"{key}.csv"
            pd.DataFrame(rows_list).to_csv(path, index=False)
            print(f"\nSaved {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
