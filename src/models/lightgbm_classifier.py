"""
LightGBM model for multi-label SDG 3 classification.

Covers Experiment 10:
  - TF-IDF sparse features are first reduced via Truncated SVD (in features.py)
    to produce dense 300-dimensional LSA vectors, then fed into LightGBM.
  - Uses OneVsRestClassifier so one LGBM model is trained per label.

LightGBM is chosen to test whether gradient-boosted trees can exploit
non-linear feature interactions that linear SVM cannot capture.
"""

import lightgbm as lgb
from sklearn.multiclass import OneVsRestClassifier

from config import SEED


def train(X_train_svd, Y_train,
          n_estimators=300, learning_rate=0.05, num_leaves=31):
    """Train OneVsRest LightGBM on SVD-reduced TF-IDF features.

    Parameters
    ----------
    X_train_svd : np.ndarray, shape (n_train, n_components)
        Dense SVD-reduced feature matrix (output of features.reduce_svd).
    Y_train : np.ndarray, shape (n_train, n_labels)
    n_estimators : int
        Number of boosting rounds.
    learning_rate : float
    num_leaves : int
        Maximum number of leaves per tree.

    Returns
    -------
    OneVsRestClassifier (fitted)
    """
    clf = OneVsRestClassifier(
        lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            random_state=SEED,
            verbose=-1,
        ),
        n_jobs=-1,
    )
    clf.fit(X_train_svd, Y_train)
    return clf
