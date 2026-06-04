"""
evaluation.py
-------------
Shared evaluation utilities used across all experiments.

Provides:
    evaluate()             - compute all five metrics for a prediction array
    tune_thresholds()      - per-label threshold tuning (minimises Hamming Loss)
    tune_thresholds_f1()   - per-label threshold tuning (maximises F1 per label)
    save_fig()             - save matplotlib figures to the outputs directory
    ExperimentTracker      - lightweight result logger with summary export
    build_eval_pipeline()  - convenience wrapper for fit-predict-evaluate
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    hamming_loss,
    f1_score,
    jaccard_score,
)


# ---------------------------------------------------------------------------
# Core metric function
# ---------------------------------------------------------------------------

def evaluate(Y_true: np.ndarray, Y_pred: np.ndarray, label: str = "") -> dict:
    """
    Compute five evaluation metrics for a multi-label prediction.

    Parameters
    ----------
    Y_true : np.ndarray, shape (n_samples, n_labels)
        Ground-truth binary label matrix.
    Y_pred : np.ndarray, shape (n_samples, n_labels)
        Predicted binary label matrix.
    label : str
        Human-readable description printed alongside the metrics.

    Returns
    -------
    dict with keys:
        Hamming Loss, F1 Micro, F1 Macro, Jaccard, Exact Match
    """
    hl  = hamming_loss(Y_true, Y_pred)
    f1m = f1_score(Y_true, Y_pred, average="micro",   zero_division=0)
    f1M = f1_score(Y_true, Y_pred, average="macro",   zero_division=0)
    jac = jaccard_score(Y_true, Y_pred, average="samples", zero_division=0)
    em  = float(np.all(Y_true == Y_pred, axis=1).mean())

    metrics = {
        "Hamming Loss": round(hl,  4),
        "F1 Micro":     round(f1m, 4),
        "F1 Macro":     round(f1M, 4),
        "Jaccard":      round(jac, 4),
        "Exact Match":  round(em,  4),
    }

    if label:
        print(f"\n{label}")
        print(f"  Hamming Loss : {hl:.4f}  (primary metric — lower is better)")
        print(f"  F1 Micro     : {f1m:.4f}")
        print(f"  F1 Macro     : {f1M:.4f}  (sensitive to rare labels)")
        print(f"  Jaccard      : {jac:.4f}")
        print(f"  Exact Match  : {em:.4f}")

    return metrics


# ---------------------------------------------------------------------------
# Threshold tuning — Hamming Loss objective
# ---------------------------------------------------------------------------

def tune_thresholds(
    scores: np.ndarray,
    Y_true: np.ndarray,
    n_steps: int = 50,
) -> np.ndarray:
    """
    Search for the per-label decision threshold that minimises each label's
    individual contribution to Hamming Loss.

    LinearSVC produces decision function scores rather than probabilities.
    The default threshold of 0 is arbitrary and sub-optimal when the class
    distribution is highly imbalanced. This function searches 50 evenly spaced
    candidates in [-1, 1] and selects the best per label.

    Parameters
    ----------
    scores : np.ndarray, shape (n_samples, n_labels)
        Decision function scores from a fitted classifier.
    Y_true : np.ndarray, shape (n_samples, n_labels)
        Ground-truth binary label matrix used to evaluate each candidate threshold.
    n_steps : int
        Number of candidate thresholds to evaluate per label.

    Returns
    -------
    np.ndarray, shape (n_labels,)
        Optimal threshold per label.
    """
    candidates = np.linspace(-1.0, 1.0, n_steps)
    n_labels   = Y_true.shape[1]
    best_thresholds = np.zeros(n_labels)

    for i in range(n_labels):
        best_hl = 1.0
        best_t  = 0.0
        for t in candidates:
            preds = (scores[:, i] >= t).astype(int)
            hl_i  = (preds != Y_true[:, i]).mean()
            if hl_i < best_hl:
                best_hl = hl_i
                best_t  = t
        best_thresholds[i] = best_t

    return best_thresholds


# ---------------------------------------------------------------------------
# Threshold tuning — F1 Macro objective
# ---------------------------------------------------------------------------

def tune_thresholds_f1(
    scores: np.ndarray,
    Y_true: np.ndarray,
    n_steps: int = 50,
) -> np.ndarray:
    """
    Search for the per-label decision threshold that maximises per-label F1.

    The Hamming-Loss-optimal threshold may suppress rare labels entirely
    (mathematically correct for HL but operationally useless). This function
    maximises F1 per label instead, accepting a small HL cost in exchange for
    improved recall on rare indicators.

    Used in Experiment 15.

    Parameters
    ----------
    scores : np.ndarray, shape (n_samples, n_labels)
        Decision function scores from a fitted classifier.
    Y_true : np.ndarray, shape (n_samples, n_labels)
        Ground-truth binary label matrix.
    n_steps : int
        Number of candidate thresholds to evaluate per label.

    Returns
    -------
    np.ndarray, shape (n_labels,)
        F1-optimal threshold per label.
    """
    candidates = np.linspace(-1.0, 1.0, n_steps)
    n_labels   = Y_true.shape[1]
    best_thresholds = np.zeros(n_labels)

    for i in range(n_labels):
        best_f1 = -1.0
        best_t  = 0.0
        for t in candidates:
            preds = (scores[:, i] >= t).astype(int)
            f1_i  = f1_score(Y_true[:, i], preds, zero_division=0)
            if f1_i > best_f1:
                best_f1 = f1_i
                best_t  = t
        best_thresholds[i] = best_t

    return best_thresholds


# ---------------------------------------------------------------------------
# Figure saving utility
# ---------------------------------------------------------------------------

def save_fig(name: str, subdir: str = "figures") -> None:
    """
    Save the current matplotlib figure to outputs/<subdir>/<name>.png.

    Creates the directory if it does not exist. Does nothing if the outputs
    directory is not available (e.g. read-only environments).

    Parameters
    ----------
    name : str
        Filename without extension.
    subdir : str
        Subdirectory under outputs/ (default: 'figures').
    """
    try:
        out_dir = os.path.join("outputs", subdir)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{name}.png")
        plt.savefig(path, bbox_inches="tight", dpi=130)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Experiment tracker
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """
    Lightweight logger that records metrics for each experiment and
    produces a summary DataFrame for display and export.

    Usage
    -----
    tracker = ExperimentTracker()
    tracker.log(1, 'TF-IDF + LR', 'baseline', metrics_dict)
    display(tracker.summary())
    """

    def __init__(self):
        self._records = []

    def log(
        self,
        exp_num: int,
        description: str,
        change: str,
        metrics: dict,
    ) -> None:
        """
        Record a completed experiment.

        Parameters
        ----------
        exp_num : int
            Experiment number (0-indexed).
        description : str
            Short description of the configuration.
        change : str
            What changed relative to the previous experiment.
        metrics : dict
            Output of evaluate().
        """
        record = {
            "Exp": exp_num,
            "Description": description,
            "Change": change,
        }
        record.update(metrics)
        self._records.append(record)

    def summary(self) -> pd.DataFrame:
        """Return all logged experiments as a sorted DataFrame."""
        if not self._records:
            return pd.DataFrame()
        df = pd.DataFrame(self._records).sort_values("Exp").reset_index(drop=True)
        return df

    def to_dataframe(self) -> pd.DataFrame:
        """Alias for summary() — backward compatibility."""
        return self.summary()

    def best(self, metric: str = "Hamming Loss", lower_is_better: bool = True) -> dict:
        """Return the record with the best value for a given metric."""
        df = self.summary()
        if df.empty:
            return {}
        if lower_is_better:
            idx = df[metric].idxmin()
        else:
            idx = df[metric].idxmax()
        return df.loc[idx].to_dict()


# ---------------------------------------------------------------------------
# TF-IDF + LinearSVC sklearn Pipeline (used for learning curves / CV)
# ---------------------------------------------------------------------------

def build_tfidf_svc_pipeline(C=1.0, max_iter=3000):
    """Return a TF-IDF + OneVsRest LinearSVC sklearn Pipeline.

    Used for learning-curve and cross-validation analysis where sklearn
    needs a single estimator that both fits and transforms.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import Pipeline
    from config import SEED, TFIDF_MAX_FEATURES, TFIDF_MIN_DF
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            sublinear_tf=True,
            min_df=TFIDF_MIN_DF,
        )),
        ('clf', OneVsRestClassifier(
            LinearSVC(C=C, max_iter=max_iter, random_state=SEED)
        )),
    ])


# Alias — notebook calls build_eval_pipeline(C=best_C) for learning curves
build_eval_pipeline = build_tfidf_svc_pipeline


# ---------------------------------------------------------------------------
# Convenience pipeline wrapper
# ---------------------------------------------------------------------------

def build_eval_pipeline(
    clf,
    X_train,
    Y_train,
    X_val,
    Y_val,
    exp_num: int,
    description: str,
    tracker: ExperimentTracker,
    use_threshold_tuning: bool = False,
) -> tuple:
    """
    Fit a classifier, predict on the validation set, evaluate, and log.

    Parameters
    ----------
    clf : sklearn estimator
        Unfitted classifier (must implement fit and predict/decision_function).
    X_train, Y_train : training features and labels.
    X_val, Y_val     : validation features and labels.
    exp_num          : experiment number for the tracker.
    description      : short description string.
    tracker          : ExperimentTracker instance.
    use_threshold_tuning : if True, use decision_function + tune_thresholds
                           instead of predict().

    Returns
    -------
    (fitted_clf, Y_pred, metrics_dict)
    """
    clf.fit(X_train, Y_train)

    if use_threshold_tuning and hasattr(clf, "decision_function"):
        scores = clf.decision_function(X_val)
        thresh = tune_thresholds(scores, Y_val)
        Y_pred = (scores >= thresh).astype(int)
    else:
        Y_pred = clf.predict(X_val)

    metrics = evaluate(Y_val, Y_pred, description)
    tracker.log(exp_num, description, "", metrics)

    return clf, Y_pred, metrics
