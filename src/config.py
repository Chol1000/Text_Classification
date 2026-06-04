"""
Central configuration for the SDG 3 Indicator Text Classification project.

All paths are relative to the repository root. The notebook's second code cell
changes the working directory to the repo root automatically, so these paths
resolve correctly whether you run locally or on Google Colab.
"""

import os

SEED     = 42
VAL_SIZE = 0.20

DATA_DIR  = 'data'
TRAIN_CSV = os.path.join(DATA_DIR, 'Devex_train.csv')
TEST_CSV  = os.path.join(DATA_DIR, 'Devex_test_questions.csv')

DIRS = {
    'figures_eda':     'outputs/figures/eda',
    'figures_results': 'outputs/figures/results',
    'models':          'outputs/models',
    'submission':      'outputs/submission',
}

TFIDF_MAX_FEATURES = 20_000
TFIDF_MIN_DF       = 2
SVD_COMPONENTS     = 300
SBERT_MODEL        = 'all-MiniLM-L6-v2'
SBERT_CHARS        = 1_000
SBERT_BATCH        = 64


def setup_dirs():
    """Create all output directories. Call once at notebook startup."""
    for path in DIRS.values():
        os.makedirs(path, exist_ok=True)
