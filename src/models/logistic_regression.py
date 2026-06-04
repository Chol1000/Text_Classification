"""
Logistic Regression model for multi-label SDG 3 classification.

Covers Experiments 1 (unigrams) and 2 (bigrams) — the baseline classifiers.
Uses OneVsRestClassifier so one binary LR model is trained per label.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from config import SEED


def train(X_train, Y_train, C=1.0, max_iter=1000):
    """Train OneVsRest Logistic Regression on pre-computed features.

    Parameters
    ----------
    X_train : array-like or sparse matrix, shape (n_train, n_features)
    Y_train : np.ndarray, shape (n_train, n_labels)
    C : float
        Inverse of regularisation strength.
    max_iter : int
        Maximum iterations for the solver.

    Returns
    -------
    OneVsRestClassifier (fitted)
    """
    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=max_iter, C=C, random_state=SEED),
        n_jobs=-1,
    )
    clf.fit(X_train, Y_train)
    return clf