"""
Feature engineering utilities for SDG 3 indicator classification.

Available builders
------------------
build_tfidf        — Fit TF-IDF on train, transform val and test.
get_sbert_model    — Load a SentenceTransformer model (call once).
encode_sbert       — Encode text arrays with a pre-loaded SBERT model.
build_type_ohe     — One-hot encode the document Type column.
reduce_svd         — Truncated SVD for dimensionality reduction.
fuse_tfidf_sbert   — L2-normalise and concatenate TF-IDF + SBERT.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer

from config import SEED, TFIDF_MAX_FEATURES, TFIDF_MIN_DF, SVD_COMPONENTS, SBERT_MODEL, SBERT_CHARS


def build_tfidf(X_train, X_val, X_test=None,
                max_features=TFIDF_MAX_FEATURES,
                ngram_range=(1, 1),
                min_df=TFIDF_MIN_DF):
    """Fit TF-IDF on training data and transform all splits.

    Returns
    -------
    vec : TfidfVectorizer (fitted)
    X_tr, X_vl : sparse matrices
    X_te : sparse matrix or None
    """
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=min_df,
    )
    X_tr = vec.fit_transform(X_train)
    X_vl = vec.transform(X_val)
    X_te = vec.transform(X_test) if X_test is not None else None
    return vec, X_tr, X_vl, X_te


def get_sbert_model(model_name=SBERT_MODEL):
    """Load and return a SentenceTransformer model.

    Load once and pass to :func:`encode_sbert` to avoid repeated disk reads.
    """
    return SentenceTransformer(model_name)


def encode_sbert(model, texts, max_chars=SBERT_CHARS, batch_size=64):
    """Encode a list of text strings with a pre-loaded SBERT model.

    Parameters
    ----------
    model : SentenceTransformer
        Pre-loaded SBERT model (from :func:`get_sbert_model`).
    texts : list[str]
        Raw or cleaned text strings.
    max_chars : int
        Truncate each text to this many characters before encoding.
    batch_size : int
        Encoding batch size.

    Returns
    -------
    np.ndarray  shape (n_samples, embedding_dim)
    """
    return model.encode(
        [t[:max_chars] for t in texts],
        batch_size=batch_size,
        show_progress_bar=True,
    )


def build_type_ohe(types_train, types_val, types_test=None):
    """One-hot encode the document Type column for each data split.

    Returns
    -------
    ohe : OneHotEncoder (fitted)
    enc_tr, enc_vl : sparse matrices
    enc_te : sparse matrix or None
    """
    ohe = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    enc_tr = ohe.fit_transform(types_train.reshape(-1, 1))
    enc_vl = ohe.transform(types_val.reshape(-1, 1))
    enc_te = ohe.transform(types_test.reshape(-1, 1)) if types_test is not None else None
    return ohe, enc_tr, enc_vl, enc_te


def reduce_svd(X_train, X_val, X_test=None, n_components=SVD_COMPONENTS):
    """Apply Truncated SVD (LSA) to a sparse feature matrix.

    Returns
    -------
    svd : TruncatedSVD (fitted)
    X_tr, X_vl : dense arrays
    X_te : dense array or None
    """
    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    X_tr = svd.fit_transform(X_train)
    X_vl = svd.transform(X_val)
    X_te = svd.transform(X_test) if X_test is not None else None
    return svd, X_tr, X_vl, X_te


def fuse_tfidf_sbert(tfidf_train, tfidf_val, sbert_train, sbert_val,
                     tfidf_test=None, sbert_test=None):
    """L2-normalise SBERT embeddings and concatenate with TF-IDF sparse matrix.

    Both feature types are normalised to prevent one modality from dominating.

    Returns
    -------
    X_tr, X_vl : sparse matrices (TF-IDF + SBERT)
    X_te : sparse matrix or None
    """
    X_tr = hstack([tfidf_train, csr_matrix(normalize(sbert_train))])
    X_vl = hstack([tfidf_val,   csr_matrix(normalize(sbert_val))])
    X_te = None
    if tfidf_test is not None and sbert_test is not None:
        X_te = hstack([tfidf_test, csr_matrix(normalize(sbert_test))])
    return X_tr, X_vl, X_te