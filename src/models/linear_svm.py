"""
models/linear_svm.py
--------------------
LinearSVC wrapper for multi-label classification via OneVsRest decomposition.

LinearSVC is well suited to sparse high-dimensional TF-IDF features because it
maximises the classification margin without computing a kernel, making it both
fast and memory-efficient on vocabularies of 20,000+ tokens (Joachims, 1998;
Hsieh et al., 2008).

Provides:
    train()    - fit a OneVsRest LinearSVC
    predict()  - predict binary labels
    sweep_c()  - sweep regularisation constant C and return the best value
"""

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import hamming_loss


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(
    X_train,
    Y_train,
    C: float = 1.0,
    class_weight=None,
    max_iter: int = 2000,
    random_state: int = 42,
) -> OneVsRestClassifier:
    """
    Fit a OneVsRest LinearSVC on the provided training data.

    OneVsRest trains one binary LinearSVC per label (Binary Relevance). Despite
    its label-independence assumption, this approach is competitive on sparse
    TF-IDF features and is the standard starting point for multi-label SVM
    classification (Tsoumakas & Katakis, 2007).

    Parameters
    ----------
    X_train : sparse or dense array, shape (n_samples, n_features)
        Training feature matrix (TF-IDF, fused TF-IDF+SBERT, etc.).
    Y_train : np.ndarray, shape (n_samples, n_labels)
        Binary multi-label target matrix.
    C : float
        Regularisation parameter. Smaller values increase regularisation
        (higher bias, lower variance). Default 1.0.
    class_weight : str or None
        Set to 'balanced' to upweight rare labels inversely proportional
        to their frequency. Increases F1 Macro but typically raises
        Hamming Loss (see Experiment 4).
    max_iter : int
        Maximum number of iterations for the solver. Increase if convergence
        warnings appear.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    OneVsRestClassifier (fitted)
    """
    base = LinearSVC(
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
    )
    clf = OneVsRestClassifier(base)
    clf.fit(X_train, Y_train)
    return clf


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def predict(clf: OneVsRestClassifier, X) -> np.ndarray:
    """
    Generate binary multi-label predictions.

    Uses the default decision threshold of 0 applied to the internal
    LinearSVC decision function. For threshold-tuned inference, use
    clf.decision_function(X) directly with tune_thresholds() from
    evaluation.py.

    Parameters
    ----------
    clf : fitted OneVsRestClassifier
    X   : feature matrix, shape (n_samples, n_features)

    Returns
    -------
    np.ndarray, shape (n_samples, n_labels), dtype int
    """
    return clf.predict(X)


# ---------------------------------------------------------------------------
# C hyperparameter sweep
# ---------------------------------------------------------------------------

def sweep_c(
    X_train,
    Y_train,
    X_val,
    Y_val,
    c_values: list = None,
    max_iter: int = 2000,
    random_state: int = 42,
) -> tuple:
    """
    Evaluate a range of C values and return results plus the best C.

    The default C=1 is arbitrary. Regularisation strength controls the
    bias-variance trade-off on sparse high-dimensional vocabularies:
    small C increases regularisation (penalises margin violations less),
    large C allows smaller margins (potentially overfitting on training data).
    This sweep selects the value that minimises validation Hamming Loss.

    Used in Experiment 5 (TF-IDF-only) and Experiment 13 (fused TF-IDF+SBERT).

    Parameters
    ----------
    X_train, Y_train : training features and labels.
    X_val, Y_val     : validation features and labels.
    c_values         : list of C values to evaluate.
                       Defaults to [0.01, 0.1, 0.5, 1.0, 5.0, 10.0].
    max_iter         : passed to LinearSVC.
    random_state     : passed to LinearSVC.

    Returns
    -------
    (sweep_df, best_C)
        sweep_df : pd.DataFrame with columns [C, Hamming Loss, F1 Micro, F1 Macro]
        best_C   : float, the C value with the lowest validation Hamming Loss
    """
    if c_values is None:
        c_values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]

    from sklearn.metrics import f1_score

    records = []
    print("LinearSVC C sweep:")
    for c in c_values:
        clf  = train(X_train, Y_train, C=c, max_iter=max_iter, random_state=random_state)
        pred = clf.predict(X_val)
        hl   = hamming_loss(Y_val, pred)
        f1m  = f1_score(Y_val, pred, average="micro",  zero_division=0)
        f1M  = f1_score(Y_val, pred, average="macro",  zero_division=0)
        records.append({"C": c, "Hamming Loss": round(hl, 4),
                        "F1 Micro": round(f1m, 4), "F1 Macro": round(f1M, 4)})
        print(f"  C={c:.3f}  HL={hl:.4f}  F1-micro={f1m:.4f}  F1-macro={f1M:.4f}")

    sweep_df = pd.DataFrame(records)
    best_C   = float(sweep_df.loc[sweep_df["Hamming Loss"].idxmin(), "C"])
    return sweep_df, best_C
