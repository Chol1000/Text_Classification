"""
Sentence-BERT + Linear SVM model for multi-label SDG 3 classification.

Covers Experiments 8 and 9:
  - Exp 8: SBERT embeddings only (all-MiniLM-L6-v2) + LinearSVC
  - Exp 9: TF-IDF + SBERT fused features + LinearSVC

SBERT embeddings are computed externally via src/features.py and passed in as
dense numpy arrays.  The classifier here is identical to linear_svm.train(),
but is kept as a separate module to explicitly represent the SBERT embedding +
classifier combination as required by the assignment.
"""

from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

from config import SEED


def train(X_train_emb, Y_train, C=1.0, max_iter=3000):
    """Train OneVsRest LinearSVC on SBERT or fused SBERT+TF-IDF embeddings.

    Parameters
    ----------
    X_train_emb : np.ndarray or sparse matrix, shape (n_train, embedding_dim)
        Pre-computed SBERT embeddings or fused (TF-IDF + SBERT) features.
    Y_train : np.ndarray, shape (n_train, n_labels)
    C : float
        Regularisation parameter.
    max_iter : int

    Returns
    -------
    OneVsRestClassifier (fitted)
    """
    clf = OneVsRestClassifier(
        LinearSVC(C=C, max_iter=max_iter, random_state=SEED)
    )
    clf.fit(X_train_emb, Y_train)
    return clf
