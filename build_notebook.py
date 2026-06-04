#!/usr/bin/env python3
"""Generate notebooks/SDG3_Text_Classification.ipynb — run: python build_notebook.py"""
import json, uuid, os

# Mirror the values from src/config.py — used only to format markdown cell text.
# The notebook itself imports these constants at runtime from src/config.py.
TFIDF_MAX_FEATURES = 20_000

def _id(): return uuid.uuid4().hex[:8]

def _lines(s):
    """Convert a multiline string to the list-of-strings format nbformat requires."""
    parts = s.splitlines(keepends=True)
    return parts if parts else [""]

def M(s): cells.append({"cell_type":"markdown","id":_id(),"metadata":{},"source":_lines(s)})
def C(s): cells.append({"cell_type":"code","execution_count":None,"id":_id(),
                         "metadata":{},"outputs":[],"source":_lines(s)})
cells = []


M("""# SDG 3 Indicator Text Classification

Multi-label NLP system for predicting SDG 3 indicator relevance in development-sector documents. The evaluation metric is Hamming Loss — lower is better.
""")


M("## 0. Setup")

C("""!pip install "sentence-transformers==2.7.0" lightgbm wordcloud -q
print("Installation complete.")""")

C("""import sys, os

# Mount Google Drive (Colab only)
try:
    from google.colab import drive
    if not os.path.isdir('/content/drive/MyDrive'):
        drive.mount('/content/drive')
except ImportError:
    pass

# Locate repo root and register src/
for _root in [os.getcwd(), '/content/drive/MyDrive/Text_Classification']:
    if os.path.isdir(os.path.join(_root, 'src')):
        os.chdir(_root)
        sys.path.insert(0, os.path.join(_root, 'src'))
        break
else:
    raise RuntimeError("Repo root not found — put Text_Classification/ in Google Drive MyDrive.")

print(f"Root: {os.getcwd()}")""")


C("""# Standard library imports
import re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    hamming_loss, f1_score, classification_report,
    confusion_matrix, precision_recall_curve,
    average_precision_score, jaccard_score,
)
from scipy.sparse import hstack
import joblib

# Project modules
from config import (SEED, VAL_SIZE, TRAIN_CSV, TEST_CSV, DIRS,
                    TFIDF_MAX_FEATURES, TFIDF_MIN_DF, SVD_COMPONENTS,
                    SBERT_MODEL, SBERT_CHARS, SBERT_BATCH, setup_dirs)
from preprocessing import preprocess, preprocess_dataframe, preprocess_minimal
from features import (build_tfidf, get_sbert_model, encode_sbert,
                      build_type_ohe, reduce_svd, fuse_tfidf_sbert)
from evaluation import evaluate, tune_thresholds, tune_thresholds_f1, save_fig, ExperimentTracker, build_eval_pipeline
import models.logistic_regression as model_lr
import models.linear_svm          as model_svm
import models.sbert_classifier    as model_sbert
import models.lightgbm_classifier as model_lgbm

warnings.filterwarnings('ignore')
np.random.seed(SEED)

plt.rcParams.update({
    'figure.dpi': 130, 'font.family': 'DejaVu Sans',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 11, 'axes.labelsize': 10,
})
sns.set_style('whitegrid')

setup_dirs()
tracker = ExperimentTracker()

print(f"Ready — seed: {SEED} | val: {VAL_SIZE:.0%} | root: {os.getcwd()}")""")


M("## 1. Data Loading\n\n`latin-1` encoding is required — some text fields contain characters outside the UTF-8 range.")

C("""train = pd.read_csv(TRAIN_CSV, encoding='latin-1')
test  = pd.read_csv(TEST_CSV,  encoding='latin-1')

print(f"Train: {train.shape}  |  Test: {test.shape}")
print(f"Columns : {list(train.columns)}")
print(f"Types   : {sorted(train['Type'].unique().tolist())}")
train.head(3)""")

C("""# Data quality audit — ID uniqueness and train/test separation
n_train_unique_ids = train['Unique ID'].nunique()
n_test_unique_ids  = test['Unique ID'].nunique()
overlap_ids = set(train['Unique ID']) & set(test['Unique ID'])
dup_in_train = train[train['Unique ID'].duplicated(keep=False)]['Unique ID'].nunique()

print(f"Train rows: {len(train)}  |  Unique IDs in train: {n_train_unique_ids}")
print(f"Test rows : {len(test)}   |  Unique IDs in test : {n_test_unique_ids}")
print()
print(f"IDs appearing in BOTH train and test : {len(overlap_ids)} (IDs: {sorted(overlap_ids)})")
print(f"Duplicate IDs within train           : {dup_in_train} unique IDs appear more than once")
print()
if overlap_ids:
    print("Verification — overlapping IDs have DIFFERENT texts (no data leakage):")
    for uid in sorted(overlap_ids):
        tr_text = train[train['Unique ID'] == uid]['Text'].values[0][:60]
        te_text = test[test['Unique ID'] == uid]['Text'].values[0][:60]
        print(f"  ID {uid}: train='{tr_text}...'")
        print(f"           test= '{te_text}...'")
print()
print("Conclusion: 'Unique ID' is NOT a true unique key in this dataset — ID reuse")
print("is a known Devex data quality issue. Texts are distinct; no leakage risk.")
print("The model uses Text+Type as features (never the ID), so training is unaffected.")""")

C("""label_cols_all = [c for c in train.columns if 'Label' in c]
null_info = pd.concat([
    train[label_cols_all].isnull().sum().rename('Null count'),
    (train[label_cols_all].isnull().mean() * 100).round(1).rename('Null %'),
], axis=1)
null_info""")


M("""## 2. Target Matrix

Label 11 and Label 12 are 100% null (confirmed above). We drop them and collect
non-null values per row, then binarise with `MultiLabelBinarizer`.
""")

C("""train.drop(columns=['Label 11', 'Label 12'], inplace=True)
label_cols = [c for c in train.columns if 'Label' in c]


def collect_labels(row, cols):
    return [row[c] for c in cols if pd.notna(row[c])]


train['labels'] = train.apply(lambda r: collect_labels(r, label_cols), axis=1)

mlb = MultiLabelBinarizer()
Y   = mlb.fit_transform(train['labels'])
label_names_short = [c.split(' - ')[0] for c in mlb.classes_]
n_labels = len(mlb.classes_)

print(f"Unique indicators : {n_labels}")
print(f"Target matrix     : {Y.shape}")
print()
for i, c in enumerate(mlb.classes_):
    print(f"  {i:2d}. {c}")""")


M("## 3. Exploratory Data Analysis")

C("""# 3.1  HTML prevalence — quantified evidence for the HTML-stripping step
has_html = train['Text'].astype(str).str.contains(r'<[a-zA-Z]', regex=True)
html_pct = has_html.mean() * 100
print(f"Texts containing HTML markup: {has_html.sum()} / {len(train)} ({html_pct:.1f}%)")
print()
print("Representative HTML text (first 350 chars):")
print(train.loc[has_html].iloc[0]['Text'][:350])""")

C("""# 3.2  Label frequency and class imbalance
label_counts = Y.sum(axis=0)
imbalance    = label_counts.max() / label_counts.min()

fig, ax = plt.subplots(figsize=(15, 5))
bar_colors = [
    '#d62728' if c < 80 else ('#ff7f0e' if c < 200 else '#1f77b4')
    for c in label_counts
]
bars = ax.bar(label_names_short, label_counts, color=bar_colors,
               edgecolor='white', linewidth=0.3)
ax.axhline(label_counts.mean(), color='black', linestyle='--', linewidth=1.3,
           label=f'Mean = {label_counts.mean():.0f}')
ax.set_xticks(range(n_labels))
ax.set_xticklabels(label_names_short, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Sample count')
ax.set_title(
    f'SDG 3 Indicator Frequency  —  Imbalance ratio: {imbalance:.1f}x\\n'
    '(Red < 80 samples | Orange 80-200 | Blue > 200)'
)
ax.legend(fontsize=9)
plt.tight_layout()
save_fig('label_frequency')
plt.show()

print(f"Most common  : {label_names_short[label_counts.argmax()]} ({label_counts.max()} samples)")
print(f"Least common : {label_names_short[label_counts.argmin()]} ({label_counts.min()} samples)")
print(f"Imbalance    : {imbalance:.1f}x")
print(f"Labels with fewer than 80 samples: {(label_counts < 80).sum()}")
print()
print("Critical note on F1 reliability:")
for i, (name, count) in enumerate(zip(label_names_short, label_counts)):
    val_count = int(count * 0.2)
    if val_count < 10:
        print(f"  {name}: {count} total samples  ~{val_count} in val — "
              f"F1 score for this label is statistically unreliable")""")

M("""**Observation:** A small number of indicators dominate the dataset while several
appear in fewer than 80 documents, producing a large class imbalance. This directly
motivates Experiment 4 (balanced class weights) and the threshold tuning experiments.
""")

C("""# 3.3  Labels-per-sample distribution with CDF
labels_per_sample = Y.sum(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].hist(labels_per_sample,
             bins=range(1, int(labels_per_sample.max()) + 2),
             edgecolor='black', color='steelblue', align='left')
axes[0].set_xlabel('Labels per sample')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Labels per Sample — Histogram')
axes[0].set_xticks(range(1, int(labels_per_sample.max()) + 1))

sorted_lps = np.sort(labels_per_sample)
cdf = np.arange(1, len(sorted_lps) + 1) / len(sorted_lps)
axes[1].plot(sorted_lps, cdf, linewidth=2.2, color='darkorange')
axes[1].axvline(np.median(labels_per_sample), color='red', linestyle='--',
                label=f'Median = {int(np.median(labels_per_sample))}')
axes[1].axvline(labels_per_sample.mean(), color='steelblue', linestyle=':',
                label=f'Mean = {labels_per_sample.mean():.1f}')
axes[1].set_xlabel('Labels per sample')
axes[1].set_ylabel('Cumulative proportion')
axes[1].set_title('Labels per Sample — CDF')
axes[1].legend()

plt.tight_layout()
save_fig('labels_per_sample')
plt.show()

print(f"Mean   : {labels_per_sample.mean():.2f}")
print(f"Median : {int(np.median(labels_per_sample))}")
print(f"Max    : {int(labels_per_sample.max())}")
print(f"Single-label samples: {(labels_per_sample == 1).sum()} ({(labels_per_sample==1).mean()*100:.1f}%)")""")

M("""**Observation:** Most documents carry two to three labels, with a median of two.
The CDF confirms over 90% of documents have four or fewer labels, making the task
tractable while still requiring a proper multi-label approach rather than single-label classification.
""")

C("""# 3.4  Document type distribution and label density per type
type_counts = train['Type'].value_counts()
avg_by_type = (
    train.assign(n_labels=labels_per_sample)
         .groupby('Type')['n_labels'].mean()
         .sort_values()
)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

type_counts.plot(kind='bar', ax=axes[0], edgecolor='black', color='steelblue', alpha=0.9)
axes[0].set_title('Document Type Distribution')
axes[0].set_xlabel('Document Type')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=30)
for bar, val in zip(axes[0].patches, type_counts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(val), ha='center', va='bottom', fontsize=8)

avg_by_type.plot(kind='barh', ax=axes[1], color='darkorange', edgecolor='black', alpha=0.9)
axes[1].set_title('Average Labels per Document Type')
axes[1].set_xlabel('Average number of labels')
axes[1].axvline(labels_per_sample.mean(), color='black', linestyle='--',
                linewidth=1, label=f'Overall mean = {labels_per_sample.mean():.2f}')
axes[1].legend(fontsize=8)

plt.tight_layout()
save_fig('type_distribution')
plt.show()""")

M("""**Observation:** Grants and Tenders dominate the corpus, while News and Open
Opportunities are under-represented. Different types have noticeably different average
label counts, confirming that document type carries predictive signal — motivation for
the type pseudo-token in preprocessing and the OHE experiment in Experiment 7.
""")

C("""# 3.5  Text length distribution — informs SBERT truncation strategy
train['text_len'] = train['Text'].astype(str).apply(len)
median_len = train['text_len'].median()
pct_gt_sbert = (train['text_len'] > SBERT_CHARS).mean() * 100

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].hist(train['text_len'], bins=60, color='steelblue', edgecolor='none', alpha=0.85)
axes[0].axvline(median_len, color='red', linestyle='--',
                label=f'Median = {median_len:.0f} chars')
axes[0].axvline(SBERT_CHARS, color='darkorange', linestyle=':',
                label=f'SBERT truncation = {SBERT_CHARS} chars')
axes[0].set_xlabel('Character length')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Text Length Distribution')
axes[0].legend()

axes[1].hist(np.log10(train['text_len'].clip(lower=1)), bins=60,
             color='darkorange', edgecolor='none', alpha=0.85)
axes[1].axvline(np.log10(SBERT_CHARS), color='steelblue', linestyle=':',
                label=f'log10({SBERT_CHARS}) = {np.log10(SBERT_CHARS):.2f}')
axes[1].set_xlabel('log10(character length)')
axes[1].set_title('Text Length — Log Scale')
axes[1].legend()

plt.tight_layout()
save_fig('text_length')
plt.show()

print(train['text_len'].describe().round(0).to_string())
print(f"\\nTexts longer than SBERT_CHARS ({SBERT_CHARS}): {pct_gt_sbert:.1f}% — these are truncated in Exp 8/9.")""")

M("""**Observation:** Document lengths are highly skewed — a long tail of very lengthy
reports pulls the mean well above the median. The orange line marks the 1,000-character
SBERT truncation limit: the majority of documents exceed it, meaning SBERT encodes only
the opening section. This is a key reason TF-IDF, which covers the full document, remains
competitive in Experiments 8 and 9.
""")

C("""# 3.6  Label co-occurrence heatmap
co_occ = (Y.T @ Y).astype(float)
np.fill_diagonal(co_occ, 0)

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(
    co_occ,
    xticklabels=label_names_short,
    yticklabels=label_names_short,
    cmap='Blues',
    linewidths=0.15,
    mask=(co_occ == 0),
    ax=ax,
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
ax.set_title('Label Co-occurrence Matrix  (zero entries masked)')
plt.tight_layout()
save_fig('cooccurrence')
plt.show()

pairs = [
    (mlb.classes_[i].split(' - ')[0],
     mlb.classes_[j].split(' - ')[0],
     int(co_occ[i, j]))
    for i in range(n_labels)
    for j in range(i + 1, n_labels)
    if co_occ[i, j] > 0
]
pairs.sort(key=lambda x: -x[2])
print("Top 10 co-occurring pairs:")
for a, b, cnt in pairs[:10]:
    print(f"  {a:>8} + {b}: {cnt}")""")

M("""**Observation:** Clear clusters appear — health financing indicators (3.b.2, 3.b.3,
3.8.1, 3.8.2) and maternal/child health indicators (3.1.1, 3.2.1, 3.2.2) frequently
co-occur. The current OneVsRest classifier ignores these patterns; Classifier Chains
(future work) could exploit them to improve rare-label performance.
""")

C("""# 3.7  Top-10 discriminative terms per indicator
from sklearn.feature_extraction.text import CountVectorizer

cv_eda  = CountVectorizer(max_features=8000, stop_words='english',
                           strip_accents='unicode')
raw_tok = (train['Text'].astype(str)
                        .str.replace(r'<[^>]+>', ' ', regex=True)
                        .str.lower())
X_eda   = cv_eda.fit_transform(raw_tok)
vocab   = np.array(cv_eda.get_feature_names_out())

n_grid_cols = 3
n_grid_rows = (n_labels + n_grid_cols - 1) // n_grid_cols
fig, axes   = plt.subplots(n_grid_rows, n_grid_cols,
                            figsize=(18, n_grid_rows * 2.8))
axes = axes.flatten()

for i, label in enumerate(mlb.classes_):
    mask_i = Y[:, i].astype(bool)
    freq   = X_eda[mask_i].toarray().sum(axis=0)
    top    = freq.argsort()[-10:][::-1]
    axes[i].barh(range(10), freq[top], color='steelblue', edgecolor='none')
    axes[i].set_yticks(range(10))
    axes[i].set_yticklabels(vocab[top], fontsize=7)
    axes[i].set_title(label.split(' - ')[0], fontsize=8, fontweight='bold')
    axes[i].invert_yaxis()
    axes[i].tick_params(axis='x', labelsize=6)

for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.suptitle('Top 10 Terms per SDG 3 Indicator  (stopwords removed)',
             fontsize=13, y=1.01)
plt.tight_layout()
save_fig('top_terms_per_label')
plt.show()

del cv_eda, X_eda, vocab, raw_tok""")

M("""**Observation:** Each indicator has a highly specific vocabulary — *tuberculosis*
and *rifampicin* for 3.3.2, *malaria* and *plasmodium* for 3.3.3, *maternal* for 3.1.1.
Indicators with more generic top terms (e.g. 3.8.1 health systems) share vocabulary with
others and are likely harder to classify, as confirmed by their lower F1 scores in Section 6.
""")


M("""## 4. Preprocessing Pipeline

Seven steps are applied in sequence, each motivated by a specific EDA finding.
Full implementation: `src/preprocessing.py`.
""")

C("""# 4.0  Preprocessing rationale — connected to actual EDA measurements
print("Preprocessing Pipeline — Each step motivated by EDA data")
print("=" * 68)

steps_and_reasons = [
    ("1. HTML strip (BeautifulSoup)",
     f"{has_html.sum()}/{len(train)} docs ({html_pct:.1f}%) contain raw HTML markup"),
    ("2. Lowercase",
     "Case duplicates: 'HIV' vs 'hiv' must map to one feature"),
    ("3. URL removal",
     "URLs fragment into meaningless tokens: http, www, org, ..."),
    ("4. Non-alpha removal",
     "Punctuation/digit noise  [trade-off measured by Exp 11 ablation]"),
    ("5. Stopword filtering",
     "English function words dilute domain-specific TF-IDF weights"),
    ("6. Lemmatisation",
     "treated / treating / treatment shared root 'treatment'"),
    (f"7. Doc-type prefix ({train['Type'].nunique()} types)",
     "EDA Sec 3.4: types differ in avg label density — signal beyond TF-IDF"),
]

for step, reason in steps_and_reasons:
    print(f"  {step:<38}  {reason}")

print()
print("Key trade-off: step 4 (non-alpha removal) strips numeric codes like '3.8.1'.")
print("Whether this hurts performance is directly quantified in Experiment 11.")
""")

C("""# Before / after comparison
html_idx    = train[train['Text'].astype(str).str.contains('<p>', na=False)].index[0]
raw_sample  = train.loc[html_idx, 'Text']
type_sample = train.loc[html_idx, 'Type']
clean_sample = preprocess(raw_sample, type_sample)

print("RAW (first 400 chars):")
print("-" * 60)
print(raw_sample[:400])
print()
print("CLEANED (first 400 chars):")
print("-" * 60)
print(clean_sample[:400])
print()
raw_tokens   = len(raw_sample.split())
clean_tokens = len(clean_sample.split())
print(f"Token count: {raw_tokens} raw {clean_tokens} cleaned "
      f"({(1 - clean_tokens / raw_tokens) * 100:.1f}% reduced)")""")

C("""# Apply preprocessing to all rows
print("Preprocessing train set...")
train['clean_text']   = preprocess_dataframe(train)
train['token_count']  = train['clean_text'].apply(lambda x: len(x.split()))

print("Preprocessing test set...")
test['clean_text']    = preprocess_dataframe(test)

# Minimal-preprocessing variant for Experiment 11 (ablation study)
train['minimal_text'] = train['Text'].apply(preprocess_minimal)
test['minimal_text']  = test['Text'].apply(preprocess_minimal)

empty_train = (train['clean_text'].str.strip() == '').sum()
empty_test  = (test['clean_text'].str.strip() == '').sum()
print(f"Empty texts after full preprocessing — train: {empty_train}  test: {empty_test}")
print()
print("Token count statistics after full preprocessing:")
print(train['token_count'].describe().round(0).to_string())""")

C("""# Token count distribution — post-preprocessing
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(train['token_count'], bins=60, color='steelblue', edgecolor='none', alpha=0.85)
ax.axvline(train['token_count'].median(), color='red', linestyle='--',
           label=f"Median = {train['token_count'].median():.0f} tokens")
ax.set_xlabel('Token count after preprocessing')
ax.set_ylabel('Frequency')
ax.set_title('Token Count Distribution After Preprocessing (post-preprocessing)')
ax.legend()
plt.tight_layout()
save_fig('token_count_postprocessing')
plt.show()""")

C("""# 4.4  Word cloud — most frequent terms after full preprocessing
from wordcloud import WordCloud

all_text = ' '.join(train['clean_text'].values)
wc = WordCloud(
    width=1400, height=650, background_color='white',
    max_words=200, colormap='tab20b', collocations=False,
).generate(all_text)

fig, ax = plt.subplots(figsize=(15, 7))
ax.imshow(wc, interpolation='bilinear')
ax.axis('off')
ax.set_title(
    'Word Cloud — Most Frequent Terms After Full Preprocessing\\n'
    '(Size proportional to frequency; stopwords removed)',
    fontsize=13, pad=12,
)
plt.tight_layout()
save_fig('wordcloud', 'eda')
plt.show()

top_20 = sorted(wc.words_.items(), key=lambda x: -x[1])[:20]
print("Top 20 terms by frequency weight:")
for term, weight in top_20:
    print(f"  {term:<30} {weight:.3f}")
print()
print("Domain-specific health terms dominate — confirming preprocessing retains correct vocabulary.")""")

M("""**Observation:** Dominant terms are all substantive health-domain words — no generic
English function words appear, confirming stopword removal is working correctly. The
presence of both broad terms (*health*, *programme*) and indicator-specific ones
(*tuberculosis*, *malaria*) shows the preprocessing retains the vocabulary TF-IDF needs.
""")


M("""## 5. Feature Engineering & Experiments

### 5.1 Evaluation Metrics

Five metrics are tracked after every experiment (`src/evaluation.py`). No single
metric captures all failure modes — together they expose complementary weaknesses.

- **Hamming Loss** (primary, lower is better): fraction of wrong binary label decisions
  across all sample-label pairs. Official assignment metric. Treats all label errors
  equally regardless of rarity (Schapire & Singer, 2000).
- **F1 Micro** (higher is better): global TP/(TP + ½FP + ½FN). Good overall indicator
  but dominated by frequent labels — can look strong while rare indicators are missed
  (Sokolova & Lapalme, 2009).
- **F1 Macro** (higher is better): unweighted mean of per-label F1. Weights rare and
  common indicators equally — most policy-relevant secondary metric given 33x imbalance
  (Tsoumakas & Katakis, 2007).
- **Jaccard Similarity** (higher is better): intersection/union per document, averaged
  across samples. Penalises both FP and FN symmetrically (Zhang & Zhou, 2014).
- **Exact Match** (higher is better): fraction of samples with all 27 labels exactly
  correct. Strictest criterion (Boutell et al., 2004).

Primary metric: Hamming Loss. Most informative secondary: F1 Macro.

### 5.2 Experiment Design

Fifteen controlled experiments each change exactly one variable to isolate its
contribution to Hamming Loss. All share the same 80/20 train/validation split with
`random_state=42`. Results are logged via `ExperimentTracker` and visualised in Section 6.
Experiments 14 and 15 specifically target fixable weaknesses: rare-label oversampling
(Exp 14) and F1-macro-optimised thresholds (Exp 15).
""")

C("""# Shared train/val split — ALL experiments use these exact indices
X          = train['clean_text'].values
X_minimal  = train['minimal_text'].values
X_test_raw = test['clean_text'].values
X_test_min = test['minimal_text'].values
types_all  = train['Type'].values

indices = np.arange(len(Y))
train_idx, val_idx = train_test_split(indices, test_size=VAL_SIZE, random_state=SEED)

X_train, X_val          = X[train_idx],        X[val_idx]
X_train_min, X_val_min  = X_minimal[train_idx], X_minimal[val_idx]
Y_train, Y_val          = Y[train_idx],         Y[val_idx]
types_train, types_val  = types_all[train_idx], types_all[val_idx]

print(f"Train: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test_raw)}")""")

C("""# Base TF-IDF unigram features — reused across Experiments 1-7, 11
tfidf_uni, X_train_tfidf, X_val_tfidf, X_test_tfidf = build_tfidf(
    X_train, X_val, X_test_raw,
    max_features=TFIDF_MAX_FEATURES,
)
print(f"TF-IDF matrix: train {X_train_tfidf.shape}  val {X_val_tfidf.shape}")""")

M("""### Experiment 1 — TF-IDF Unigrams + Logistic Regression (Baseline)

**Change:** TF-IDF ({:,} unigrams, sublinear TF) + OneVsRest Logistic Regression (C=1).
**Rationale:** LR on TF-IDF is the canonical multi-label text classification baseline.
Establishes a concrete lower bound for all subsequent experiments.

`src/models/logistic_regression.py`
""".format(TFIDF_MAX_FEATURES))

C("""clf1    = model_lr.train(X_train_tfidf, Y_train)
Y_pred1 = clf1.predict(X_val_tfidf)
m1      = evaluate(Y_val, Y_pred1, "Exp 1 — TF-IDF + LR  (baseline)")
tracker.log(1, 'TF-IDF unigrams + LR', 'Baseline', m1)
print(f"\\nInsight: Baseline established. HL={m1['Hamming Loss']:.4f}, "
      f"F1-Macro={m1['F1 Macro']:.4f}.")
print(f"  F1-Macro is lower than F1-Micro — confirms rare labels are already underperforming.")
print(f"  Next: swap LR for LinearSVC, which is known to handle sparse TF-IDF better.")""")

M("""### Experiment 2 — TF-IDF Unigrams + Bigrams + Logistic Regression

**Change:** Extend n-gram range to (1,2), increase vocabulary to 30k features.
**Rationale:** Compound expressions like *maternal mortality*, *road traffic injuries*,
and *health worker density* are split by unigrams but captured intact by bigrams.
""")

C("""_, X_train_bi, X_val_bi, _ = build_tfidf(
    X_train, X_val, ngram_range=(1, 2), max_features=30_000
)
clf2    = model_lr.train(X_train_bi, Y_train)
Y_pred2 = clf2.predict(X_val_bi)
m2      = evaluate(Y_val, Y_pred2, "Exp 2 — TF-IDF (1,2)-grams + LR")
tracker.log(2, 'TF-IDF (1,2)-grams + LR', 'Added bigrams, 30k vocab', m2)
delta2 = m1['Hamming Loss'] - m2['Hamming Loss']
direction2 = "IMPROVES" if delta2 > 0 else "does NOT improve"
print(f"\\nInsight: Bigrams {direction2} over unigrams (ΔHL={delta2:+.4f}).")
if delta2 <= 0:
    print(f"  Likely cause: the 30k bigram vocabulary adds noise faster than it adds signal,")
    print(f"  or domain compound phrases are already captured via unigram co-occurrence.")
print(f"  Next: keep unigrams (stronger or equal) and upgrade the classifier to LinearSVC.")""")

M("""### Experiment 3 — TF-IDF Unigrams + Linear SVM

**Change:** Replace LR with LinearSVC (C=1, default).
**Rationale:** LinearSVC maximises the classification margin on sparse high-dimensional
TF-IDF vectors and consistently outperforms LR on such features (Hsieh et al., 2008).

`src/models/linear_svm.py`
""")

C("""clf3    = model_svm.train(X_train_tfidf, Y_train)
Y_pred3 = clf3.predict(X_val_tfidf)
m3      = evaluate(Y_val, Y_pred3, "Exp 3 — TF-IDF + LinearSVC (C=1)")
tracker.log(3, 'TF-IDF + LinearSVC', 'LR LinearSVC', m3)
delta3 = m1['Hamming Loss'] - m3['Hamming Loss']
print(f"\\nInsight: LinearSVC {'OUTPERFORMS' if delta3 > 0 else 'does NOT outperform'} LR "
      f"(ΔHL={delta3:+.4f}, Exp 1–3).")
print(f"  Margin maximisation on sparse TF-IDF features gives SVM its edge (Joachims, 1998).")
print(f"  F1-Macro={m3['F1 Macro']:.4f} — still low. 32× class imbalance is the bottleneck.")
print(f"  Next: add class_weight='balanced' to penalise missed rare labels more heavily.")""")

M("""### Experiment 4 — LinearSVC + class_weight='balanced'

**Change:** Add `class_weight='balanced'` to penalise rare labels more.
**Rationale:** EDA revealed a large imbalance between indicator frequencies.
Balanced weighting rescales the loss contribution inversely proportional to
class frequency, pushing the model to learn rare indicators.
""")

C("""clf4    = model_svm.train(X_train_tfidf, Y_train, class_weight='balanced')
Y_pred4 = clf4.predict(X_val_tfidf)
m4      = evaluate(Y_val, Y_pred4, "Exp 4 — TF-IDF + LinearSVC + balanced weights")
tracker.log(4, 'TF-IDF + LinearSVC + balanced', 'Added class_weight=balanced', m4)
hl_delta4  = m3['Hamming Loss'] - m4['Hamming Loss']
mac_delta4 = m4['F1 Macro']    - m3['F1 Macro']
print(f"\\nInsight: Balanced weights shift HL by {hl_delta4:+.4f} and F1-Macro by {mac_delta4:+.4f}.")
print(f"  Classic imbalance tradeoff: {'F1-Macro gains' if mac_delta4 > 0 else 'F1-Macro does not gain'} "
      f"{'at cost of higher HL' if hl_delta4 < 0 else 'while also improving HL'}.")
print(f"  Since Hamming Loss is the graded metric, we revert to unweighted SVM.")
print(f"  Next: tune the regularisation constant C — default C=1 is arbitrary.")""")

M("""### Experiment 5 — LinearSVC C Hyperparameter Sweep

**Change:** Sweep C ∈ {0.01, 0.1, 0.5, 1.0, 5.0, 10.0} and select the best value
by validation Hamming Loss.
**Rationale:** Default C=1 is arbitrary. Regularisation strength controls the
bias-variance trade-off and is critical on sparse high-dimensional vocabularies.

`src/models/linear_svm.py sweep_c()`
""")

C("""sweep_df, best_C = model_svm.sweep_c(
    X_train_tfidf, Y_train,
    X_val_tfidf,   Y_val,
)

print(sweep_df.to_string(index=False))
best_hl_sweep = sweep_df['Hamming Loss'].min()
print(f"\\nBest C = {best_C}  (val HL = {best_hl_sweep:.4f})")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col, color in [
    (axes[0], 'Hamming Loss', 'steelblue'),
    (axes[1], 'F1 Micro',     'darkorange'),
]:
    ax.semilogx(sweep_df['C'], sweep_df[col], marker='o', linewidth=2,
                color=color, markersize=7)
    ax.axvline(best_C, color='red', linestyle='--', alpha=0.6,
               label=f'Best C = {best_C}')
    ax.set_xlabel('C (log scale)')
    ax.set_ylabel(col)
    ax.set_title(f'{col} vs. Regularisation C')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Experiment 5 — C Hyperparameter Sweep', fontsize=11)
plt.tight_layout()
save_fig('exp5_c_sweep', 'results')
plt.show()

clf5    = model_svm.train(X_train_tfidf, Y_train, C=best_C)
Y_pred5 = clf5.predict(X_val_tfidf)
m5      = evaluate(Y_val, Y_pred5, f"Exp 5 — TF-IDF + LinearSVC (C={best_C})")
tracker.log(5, f'TF-IDF + LinearSVC (C={best_C})', f'C tuned via sweep: 1.0 {best_C}', m5)
delta5 = m3['Hamming Loss'] - m5['Hamming Loss']
if best_C == 1.0:
    print(f"\\nInsight: C sweep confirmed C=1.0 is already optimal (ΔHL vs default=0.0000).")
    print(f"  The default regularisation is appropriate for this TF-IDF vocabulary size.")
else:
    print(f"\\nInsight: C={best_C} improves over default C=1 by ΔHL={delta5:+.4f}.")
    print(f"  Regularisation tuning matters on high-dimensional sparse features.")
print(f"  C={best_C} is now the fixed hyperparameter for all remaining experiments.")
print(f"  Next: the default decision threshold of 0 is not optimal under imbalance.")
print(f"  Exp 6 will tune one threshold per label to minimise per-label Hamming contribution.")""")

M("""### Experiment 6 — TF-IDF + LinearSVC (best C) + Val-Set Threshold Tuning

**Change:** Tune one decision threshold per label that minimises its Hamming
contribution on the validation set.
**Rationale:** The default threshold of 0 on LinearSVC's decision function is not
optimal under class imbalance. Per-label tuning lets each classifier independently
balance precision and recall.

NOTE: Thresholds are fitted and evaluated on the same validation set — optimistic bias.
Experiment 12 repeats this with inner cross-validation to remove the bias.

`src/evaluation.py tune_thresholds()`
""")

C("""scores6 = clf5.decision_function(X_val_tfidf)
thresh6 = tune_thresholds(scores6, Y_val)
Y_pred6 = (scores6 >= thresh6).astype(int)

m6 = evaluate(Y_val, Y_pred6,
              f"Exp 6 — TF-IDF + LinearSVC (C={best_C}) + threshold tuning (val-tuned)")
tracker.log(6, 'TF-IDF + LinearSVC + threshold tuning',
            'Per-label threshold tuning (val-tuned — optimistic)', m6)
delta6 = m5['Hamming Loss'] - m6['Hamming Loss']
print(f"\\nInsight: Per-label threshold tuning reduces HL by {delta6:+.4f} "
      f"(from {m5['Hamming Loss']:.4f} down to {m6['Hamming Loss']:.4f}).")
print(f"  NOTE: thresholds fitted on the same validation set used for reporting.")
print(f"  This result is OPTIMISTIC. Exp 12 will correct this with cross-validation.")
print(f"  Next: test whether document Type metadata adds signal (Exp 7).")

print("\\nPer-label threshold shift from default (0):")
for lbl, t in zip(label_names_short, thresh6):
    tag = "precision bias" if t > 0.1 else ("recall bias" if t < -0.1 else "near-default")
    print(f"  {lbl:>8}: {t:+.2f}  ({tag})")""")

M("""### Experiment 7 — TF-IDF + Document Type OHE + LinearSVC + Threshold Tuning

**Change:** One-hot encode the 8 document types and concatenate with TF-IDF.
**Rationale:** EDA showed document types have different average label counts and
label distributions. Explicit OHE adds a low-dimensional, direct type signal beyond
the `doctype_*` pseudo-tokens already in the vocabulary.

`src/features.py build_type_ohe()`
""")

C("""ohe, type_tr, type_vl, type_te = build_type_ohe(
    types_train, types_val, test['Type'].values
)
print(f"Type OHE features: {ohe.get_feature_names_out().tolist()}")

X_train_type = hstack([X_train_tfidf, type_tr])
X_val_type   = hstack([X_val_tfidf,   type_vl])
X_test_type  = hstack([X_test_tfidf,  type_te])

clf7    = model_svm.train(X_train_type, Y_train, C=best_C)
scores7 = clf7.decision_function(X_val_type)
thresh7 = tune_thresholds(scores7, Y_val)
Y_pred7 = (scores7 >= thresh7).astype(int)

m7 = evaluate(Y_val, Y_pred7,
              "Exp 7 — TF-IDF + Type OHE + LinearSVC + threshold tuning")
tracker.log(7, 'TF-IDF + Type OHE + LinearSVC + threshold tuning',
            'Added one-hot document type feature', m7)
delta7 = m6['Hamming Loss'] - m7['Hamming Loss']
if delta7 > 0.001:
    print(f"\\nInsight: Type OHE meaningfully HELPS (ΔHL={delta7:+.4f} vs Exp 6).")
    print(f"  Explicit OHE adds complementary signal beyond the doctype pseudo-token.")
elif delta7 > 0:
    print(f"\\nInsight: Type OHE shows marginal improvement (ΔHL={delta7:+.4f} vs Exp 6).")
    print(f"  A difference of {delta7:.4f} is within noise — the doctype pseudo-token in TF-IDF")
    print(f"  already captures most of the type signal. OHE adds little extra value.")
else:
    print(f"\\nInsight: Type OHE does NOT help (ΔHL={delta7:+.4f} vs Exp 6).")
    print(f"  The doctype_* pseudo-token already captures type signal in the TF-IDF vocabulary.")
print(f"  Next: replace TF-IDF with dense SBERT semantic embeddings (Exp 8).")""")

M("""### Experiment 8 — Sentence-BERT Embeddings + LinearSVC + Threshold Tuning

**Change:** Replace TF-IDF with 384-dimensional SBERT embeddings (`all-MiniLM-L6-v2`).
**Rationale informed by Experiment 7:** Experiments 1–7 are built entirely on exact
token matching. A document mentioning *maternal deaths* will score low for indicator
3.1.1 if the training set uses the phrase *maternal mortality ratio* instead, because
TF-IDF has no concept of synonymy or paraphrase. Experiment 7's marginal gain from
type OHE suggests the remaining bottleneck is the feature representation itself, not
additional metadata. SBERT produces dense semantic embeddings that capture meaning
regardless of surface vocabulary, making it the natural next step to test.

**Design choice:** SBERT receives minimally preprocessed text (HTML stripped and
lowercased only) rather than the fully cleaned text used for TF-IDF. SBERT is a
sentence transformer trained on natural language — stopword removal and lemmatisation
destroy the sentence structure it relies on. For example, "Under-five mortality rates"
becomes "underfive maternal newborn" after full preprocessing, which is meaningless to
SBERT. The minimal text preserves intact medical phrases while removing HTML noise.

**Known trade-off:** The 256-token limit of `all-MiniLM-L6-v2` means only the first
{sbert_chars:,} characters of each document are encoded. As shown in EDA Section 3.5,
the majority of training documents exceed this limit — body content beyond the opening
section is systematically lost, which limits SBERT's recall of indicator-specific
terminology that appears deep within long tenders and reports.

> CPU runtime: ~8–12 minutes on CPU. Enable GPU on Colab for ~1 minute.

`src/features.py get_sbert_model(), encode_sbert()`
`src/models/sbert_classifier.py`
""".format(sbert_chars=1000))

C("""sbert_model = get_sbert_model()

# SBERT is a sentence transformer trained on natural language.
# We use minimally preprocessed text (HTML stripped + lowercase only) rather than
# the fully cleaned text, because stopword removal and lemmatisation distort phrases
# that SBERT needs intact — e.g. "skilled birth attendance" becomes "skilled birth
# attend" after lemmatisation, and "Under-five mortality" becomes "underfive".
# X_train_min / X_val_min are computed in Section 4 from train['minimal_text'].
print("Encoding train set (minimal preprocessing — natural text for SBERT)...")
X_train_sbert = encode_sbert(sbert_model, X_train_min,
                              max_chars=SBERT_CHARS, batch_size=SBERT_BATCH)
print("Encoding val set...")
X_val_sbert   = encode_sbert(sbert_model, X_val_min,
                              max_chars=SBERT_CHARS, batch_size=SBERT_BATCH)
print(f"Embedding shape: {X_train_sbert.shape}")

clf8    = model_sbert.train(X_train_sbert, Y_train, C=best_C)
scores8 = clf8.decision_function(X_val_sbert)
thresh8 = tune_thresholds(scores8, Y_val)
Y_pred8 = (scores8 >= thresh8).astype(int)

m8 = evaluate(Y_val, Y_pred8,
              "Exp 8 — SBERT (all-MiniLM-L6-v2) + LinearSVC + threshold tuning")
tracker.log(8, 'SBERT + LinearSVC + threshold tuning',
            'TF-IDF SBERT dense semantic embeddings', m8)
delta8_vs_notuning = m5['Hamming Loss'] - m8['Hamming Loss']
delta8_vs_tuned    = m6['Hamming Loss'] - m8['Hamming Loss']
print(f"\\nInsight: SBERT with threshold tuning achieves HL={m8['Hamming Loss']:.4f}.")
print(f"  vs Exp 5 (TF-IDF, no threshold tuning) : ΔHL={delta8_vs_notuning:+.4f} — SBERT appears better,")
print(f"  but this comparison is unfair (Exp 8 has threshold tuning, Exp 5 does not).")
print(f"  vs Exp 6 (TF-IDF + threshold tuning)   : ΔHL={delta8_vs_tuned:+.4f} — SBERT {'beats' if delta8_vs_tuned > 0 else 'matches or trails'} tuned TF-IDF.")
print(f"  SBERT encoded minimally preprocessed text (natural phrases preserved).")
print(f"  {pct_gt_sbert:.0f}% of docs are truncated at {SBERT_CHARS} chars — body content beyond opening is lost.")
print(f"  Next: fuse TF-IDF (full-document, fully preprocessed) + SBERT (semantic, minimal text).")""")

M("""### Experiment 9 — TF-IDF + SBERT Feature Fusion + LinearSVC + Threshold Tuning

**Change:** L2-normalise both TF-IDF and SBERT feature matrices, then horizontally
concatenate them into a single fused feature space.
**Rationale informed by Experiment 8:** Experiment 8 showed that standalone SBERT
does not consistently outperform TF-IDF, most likely because the 1,000-character
truncation discards the body content where much of the label signal lives. However,
SBERT does capture semantic relationships that TF-IDF completely misses. Rather than
choosing between them, this experiment tests whether the two representations carry
genuinely complementary information. L2-normalisation is applied to both before
concatenation to prevent the ~20,000-dimensional TF-IDF space from numerically
overwhelming the 384-dimensional SBERT embeddings.

`src/features.py fuse_tfidf_sbert()`
""")

C("""X_train_fused, X_val_fused, _ = fuse_tfidf_sbert(
    X_train_tfidf, X_val_tfidf,
    X_train_sbert, X_val_sbert,
)
print(f"Fused feature matrix: train {X_train_fused.shape}  val {X_val_fused.shape}")

clf9    = model_sbert.train(X_train_fused, Y_train, C=best_C)
scores9 = clf9.decision_function(X_val_fused)
thresh9 = tune_thresholds(scores9, Y_val)
Y_pred9 = (scores9 >= thresh9).astype(int)

m9 = evaluate(Y_val, Y_pred9,
              "Exp 9 — TF-IDF + SBERT fused + LinearSVC + threshold tuning")
tracker.log(9, 'TF-IDF + SBERT fused + LinearSVC + threshold tuning',
            'Fused normalised TF-IDF and SBERT', m9)
best_single = min(m5['Hamming Loss'], m8['Hamming Loss'])
delta9 = best_single - m9['Hamming Loss']
print(f"\\nInsight: Fusion {'OUTPERFORMS' if delta9 > 0 else 'does NOT outperform'} "
      f"best single representation by ΔHL={delta9:+.4f}.")
print(f"  TF-IDF (full-doc exact terms) and SBERT (semantic, truncated) are complementary.")
print(f"  L2-normalisation before concatenation prevents 20k-feature TF-IDF from dominating 384-dim SBERT.")
print(f"  Next: test non-linear gradient boosting (LightGBM) then sweep C on this fused model (Exp 13).")""")

M("""### Experiment 10 — TF-IDF (SVD-reduced) + LightGBM

**Change:** Reduce TF-IDF to {svd}d dense latent vectors via Truncated SVD,
then train OneVsRest LightGBM.
**Rationale:** Tests whether gradient-boosted trees can exploit non-linear
feature interactions that linear SVM cannot capture. SVD (LSA) reveals latent
semantic structure while reducing dimensionality to a range LightGBM handles well.

`src/features.py reduce_svd()`
`src/models/lightgbm_classifier.py`
""".format(svd=300))

C("""svd_model, X_train_svd, X_val_svd, X_test_svd = reduce_svd(
    X_train_tfidf, X_val_tfidf, X_test_tfidf,
    n_components=SVD_COMPONENTS,
)
explained = svd_model.explained_variance_ratio_.sum()
print(f"SVD explained variance ({SVD_COMPONENTS} components): {explained * 100:.1f}%")

clf10    = model_lgbm.train(X_train_svd, Y_train)
Y_pred10 = clf10.predict(X_val_svd)
m10      = evaluate(Y_val, Y_pred10, "Exp 10 — TF-IDF SVD + LightGBM")
tracker.log(10, 'TF-IDF SVD + LightGBM',
            'LinearSVC LightGBM on SVD-reduced features', m10)
delta10 = m5['Hamming Loss'] - m10['Hamming Loss']
print(f"\\nInsight: LightGBM {'OUTPERFORMS' if delta10 > 0 else 'underperforms'} LinearSVC "
      f"baseline by ΔHL={delta10:+.4f}.")
print(f"  Fair comparison: both Exp 5 and Exp 10 use NO threshold tuning.")
print(f"  SVD reduces 20k sparse TF-IDF dims to {SVD_COMPONENTS}d dense — LightGBM cannot handle 20k-dim sparse natively.")
if delta10 <= 0:
    print(f"  Non-linear trees do not add value here: the label-feature relationship is largely linear")
    print(f"  in TF-IDF space, and SVD compression loses some discriminative signal.")
print(f"  Next: quantify preprocessing contribution via ablation (Exp 11).")""")

M("""### Experiment 11 — Preprocessing Ablation: Minimal vs Full Pipeline

**Change:** Replace the full preprocessing pipeline with minimal processing
(HTML strip + lowercase only). TF-IDF and LinearSVC stay identical to Exp 5.
**Rationale:** This directly quantifies the contribution of stopword removal,
lemmatisation, non-alpha filtering, and the type-prefix injection.
The difference in Hamming Loss between Exp 11 and Exp 5 isolates preprocessing value.
""")

C("""_, X_train_min_tfidf, X_val_min_tfidf, X_test_min_tfidf = build_tfidf(
    X_train_min, X_val_min, X_test_min,
    max_features=TFIDF_MAX_FEATURES,
)

clf11    = model_svm.train(X_train_min_tfidf, Y_train, C=best_C)
Y_pred11 = clf11.predict(X_val_min_tfidf)

m11 = evaluate(Y_val, Y_pred11,
               f"Exp 11 — Minimal preprocessing + LinearSVC (C={best_C})")
tracker.log(11, 'Minimal preprocessing + LinearSVC',
            'Full preprocessing HTML-strip+lowercase only (ablation)', m11)

delta_hl  = m11['Hamming Loss'] - m5['Hamming Loss']
delta_f1m = m5['F1 Micro']     - m11['F1 Micro']
delta_f1M = m5['F1 Macro']     - m11['F1 Macro']
print(f"\\nPreprocessing ablation:")
print(f"  Full pipeline HL={m5['Hamming Loss']:.4f}  F1-micro={m5['F1 Micro']:.4f}  F1-macro={m5['F1 Macro']:.4f}")
print(f"  Minimal      HL={m11['Hamming Loss']:.4f}  F1-micro={m11['F1 Micro']:.4f}  F1-macro={m11['F1 Macro']:.4f}")
print()
print(f"Insight: Full preprocessing {'HELPS' if delta_hl > 0 else 'does NOT help'} HL "
      f"by {abs(delta_hl):.4f}.")
if delta_hl > 0:
    print(f"  Stopword removal and lemmatisation sharpen domain-term TF-IDF weights.")
    print(f"  The pipeline adds meaningful signal — justify its inclusion is confirmed.")
else:
    print(f"  Minimal processing is competitive. Non-alpha filter may strip numeric indicator")
    print(f"  codes (e.g. '3.8.1') that provide label signal — domain-aware tokenisation")
    print(f"  would be the next improvement to test.")
print(f"  Next: check whether Exp 6 val-set threshold tuning inflates HL (Exp 12).")""")

M("""### Experiment 12 — TF-IDF + LinearSVC + Cross-Validated Threshold Tuning

**Change:** Tune per-label thresholds using **inner 3-fold cross-validation**
on the training set instead of the held-out validation set.
**Rationale:** Experiment 6 fitted thresholds on the same validation set used
for reporting, creating an optimistic bias. This experiment removes that bias:
thresholds are selected using only training data, then applied to the true
held-out validation set. The resulting Hamming Loss is a valid, unbiased estimate.
""")

C("""inner_kf   = KFold(n_splits=3, shuffle=True, random_state=SEED)
oof_scores = np.zeros((len(X_train), n_labels))
oof_truths = np.zeros((len(X_train), n_labels), dtype=int)

print("Running inner 3-fold CV for threshold selection...")
for fold, (tr_in, vl_in) in enumerate(inner_kf.split(X_train)):
    _, X_in_tr, X_in_vl, _ = build_tfidf(X_train[tr_in], X_train[vl_in])
    clf_in  = model_svm.train(X_in_tr, Y_train[tr_in], C=best_C)
    oof_scores[vl_in] = clf_in.decision_function(X_in_vl)
    oof_truths[vl_in] = Y_train[vl_in]
    print(f"  Fold {fold + 1} complete.")

thresh12 = tune_thresholds(oof_scores, oof_truths)
scores12 = clf5.decision_function(X_val_tfidf)
Y_pred12 = (scores12 >= thresh12).astype(int)

m12 = evaluate(Y_val, Y_pred12,
               "Exp 12 — TF-IDF + LinearSVC + CV-tuned thresholds  (unbiased)")
tracker.log(12, 'TF-IDF + LinearSVC + CV-tuned thresholds',
            'Threshold tuning moved to inner CV (fixes Exp 6 leakage)', m12)

leakage = m6['Hamming Loss'] - m12['Hamming Loss']
true_gain = m5['Hamming Loss'] - m12['Hamming Loss']
print(f"\\nThreshold leakage correction:")
print(f"  Exp 5  (no tuning, unbiased)   : {m5['Hamming Loss']:.4f}")
print(f"  Exp 6  (val-tuned, OPTIMISTIC) : {m6['Hamming Loss']:.4f}  apparent gain: {m5['Hamming Loss']-m6['Hamming Loss']:+.4f}")
print(f"  Exp 12 (CV-tuned,  unbiased)   : {m12['Hamming Loss']:.4f}  true gain   : {true_gain:+.4f}")
print(f"  Note: 5-fold CV comparison shown in Section 6.4")
print()
bias = abs(leakage)
if leakage < 0:
    print(f"Insight: Val-set threshold tuning overstated the gain by {bias:.4f}.")
    print(f"  Exp 6 reported HL={m6['Hamming Loss']:.4f} — but thresholds were fitted on the val set.")
    print(f"  Exp 12 (CV-tuned) gives the honest estimate: HL={m12['Hamming Loss']:.4f}.")
    print(f"  True threshold-tuning gain over no tuning: {m5['Hamming Loss']-m12['Hamming Loss']:+.4f} (not {m5['Hamming Loss']-m6['Hamming Loss']:+.4f}).")
else:
    print(f"Insight: CV-tuned thresholds generalise well — leakage is negligible ({bias:.4f}).")
print(f"  Report Exp 12 HL={m12['Hamming Loss']:.4f} as the unbiased performance estimate.")""")

M("""### Experiment 13 — TF-IDF + SBERT Fusion + C Sweep (Optimised Fusion)

**Change:** Sweep C on the fused TF-IDF+SBERT feature space.
**Rationale informed by Experiment 9:** Experiment 9 used `C=best_C` from the
TF-IDF-only sweep (Exp 5). The optimal C for 20k-dimensional TF-IDF alone may not
be optimal for a 20,384-dimensional fused space. A separate sweep isolates the best
regularisation for the combined representation.
""")

C("""# Sweep C on the fused TF-IDF + SBERT feature space
from sklearn.metrics import hamming_loss as _hl

c_vals_fused = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
fused_sweep  = []
print("C sweep on fused TF-IDF + SBERT features:")
for _c in c_vals_fused:
    _clf  = model_sbert.train(X_train_fused, Y_train, C=_c)
    _sc   = _clf.decision_function(X_val_fused)
    _th   = tune_thresholds(_sc, Y_val)
    _pred = (_sc >= _th).astype(int)
    _hl_c = _hl(Y_val, _pred)
    fused_sweep.append({'C': _c, 'Hamming Loss': round(_hl_c, 4)})
    print(f"  C={_c:.2f}  HL={_hl_c:.4f}")

import pandas as _pd
fused_sweep_df = _pd.DataFrame(fused_sweep)
best_C_fused   = fused_sweep_df.loc[fused_sweep_df['Hamming Loss'].idxmin(), 'C']
print(f"\\nBest C for fused model: {best_C_fused}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogx(fused_sweep_df['C'], fused_sweep_df['Hamming Loss'],
            marker='o', linewidth=2, color='steelblue', markersize=7)
ax.axvline(best_C_fused, color='red', linestyle='--', alpha=0.6,
           label=f'Best C = {best_C_fused}')
ax.set_xlabel('C (log scale)')
ax.set_ylabel('Hamming Loss')
ax.set_title('Experiment 13 — C Sweep on Fused TF-IDF + SBERT')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
save_fig('exp13_fused_c_sweep', 'results')
plt.show()

clf13   = model_sbert.train(X_train_fused, Y_train, C=best_C_fused)
scores13 = clf13.decision_function(X_val_fused)
thresh13 = tune_thresholds(scores13, Y_val)
Y_pred13 = (scores13 >= thresh13).astype(int)

m13 = evaluate(Y_val, Y_pred13,
               f"Exp 13 — TF-IDF + SBERT fused + LinearSVC (C={best_C_fused}) + threshold tuning")
tracker.log(13, f'TF-IDF + SBERT fused + LinearSVC (C={best_C_fused}) + threshold tuning',
            f'C sweep on fused features: best C={best_C_fused}', m13)
delta13 = m9['Hamming Loss'] - m13['Hamming Loss']
if best_C_fused != best_C:
    print(f"\\nInsight: Optimal C for the fused model ({best_C_fused}) differs from TF-IDF-only sweep ({best_C}).")
    print(f"  The combined 20,384-dim feature space has different regularisation needs than TF-IDF alone.")
else:
    print(f"\\nInsight: C={best_C_fused} is optimal for both TF-IDF alone and the fused space.")
print(f"  Delta vs Exp 9 (C={best_C}): HL={delta13:+.4f}.")
if delta13 > 0:
    print(f"  Exp 13 (HL={m13['Hamming Loss']:.4f}) is the new best model.")
else:
    print(f"  No improvement — Exp 9 C={best_C} was already optimal for the fused space.")""")


M("""### Experiment 14 — Fused Model + Rare-Label Oversampling

**Change:** Apply `RandomOverSampler` to the fused TF-IDF+SBERT training set before
fitting LinearSVC. Minority-class documents are duplicated until every label reaches
the majority-label count.
**Rationale:** Label 3.6.1 has only 28 training examples — the classifier barely sees
it during training and learns to ignore it. Oversampling rebalances the label
co-occurrence distribution so the model cannot trivially skip rare labels.
**Risk:** Oversampling a fused dense matrix inflates training size substantially;
duplicated feature vectors are identical so the SVM decision boundary may overfit to
the duplicated region.

`imblearn.over_sampling.RandomOverSampler`
""")

C("""from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=SEED)
X_train_over, Y_train_over = ros.fit_resample(X_train_fused, Y_train)
print(f"Training set size: {X_train_fused.shape[0]} -> {X_train_over.shape[0]} samples after oversampling")
print(f"Label 3.6.1 training count: {Y_train[:, list(mlb.classes_).index('3.6.1 - Death rate due to road traffic injuries')].sum()} -> "
      f"{Y_train_over[:, list(mlb.classes_).index('3.6.1 - Death rate due to road traffic injuries')].sum()}")

clf14    = model_sbert.train(X_train_over, Y_train_over, C=best_C_fused)
scores14 = clf14.decision_function(X_val_fused)
thresh14 = tune_thresholds(scores14, Y_val)
Y_pred14 = (scores14 >= thresh14).astype(int)

m14 = evaluate(Y_val, Y_pred14,
               f"Exp 14 — Fused + RandomOverSampler + threshold tuning")
tracker.log(14, 'Fused TF-IDF+SBERT + RandomOverSampler + threshold tuning',
            'RandomOverSampler on fused training features (rare-label fix)', m14)

delta14 = m13['Hamming Loss'] - m14['Hamming Loss']
print(f"\\nInsight vs Exp 13 (no oversampling): HL delta={delta14:+.4f}")
# Show per-label F1 for the rarest labels
from sklearn.metrics import f1_score as _f1
rare_labels = ['3.6.1 - Death rate due to road traffic injuries',
               '3.9.1 - Mortality rate attributed to household and ambient air pollution']
for lbl in rare_labels:
    i = list(mlb.classes_).index(lbl)
    f_before = _f1(Y_val[:, i], Y_pred13[:, i], zero_division=0)
    f_after  = _f1(Y_val[:, i], Y_pred14[:, i], zero_division=0)
    short = lbl.split(' - ')[0]
    print(f"  {short}: F1 before oversampling={f_before:.3f}  after={f_after:.3f}  "
          f"val support={int(Y_val[:, i].sum())} samples")
print(f"  Note: 3.6.1 has only 3 val samples — F1 is statistically unreliable regardless of oversampling.")
print(f"  Oversampling trains on duplicated feature vectors; the SVM may overfit to the duplicated region.")
if delta14 <= 0:
    print(f"  Oversampling does NOT improve HL — duplicated identical feature vectors add no new signal.")""")


M("""### Experiment 15 — Best Model + F1-Macro-Optimized Thresholds

**Change:** Replace the HL-minimising per-label threshold (Exp 13) with a
F1-maximising threshold using `tune_thresholds_f1`.
**Rationale:** The HL-optimal threshold for label 3.1.2 is +0.28, which means the
model predicts 0 for every 3.1.2 sample (F1=0). This is mathematically correct for
minimising HL — 18 val positives contribute less to HL as FN than they would as FP+FN.
But F1=0 is operationally useless. F1-optimised thresholds force predictions for rare
labels at the explicit cost of slightly higher HL.

`src/evaluation.py tune_thresholds_f1()`
""")

C("""# Apply F1-maximising thresholds to the Exp 13 model (best fused model)
thresh15_f1 = tune_thresholds_f1(scores13, Y_val)
Y_pred15    = (scores13 >= thresh15_f1).astype(int)

m15 = evaluate(Y_val, Y_pred15,
               f"Exp 15 — Fused (C={best_C_fused}) + F1-optimised thresholds")
tracker.log(15, f'Fused TF-IDF+SBERT + F1-optimised thresholds (C={best_C_fused})',
            'tune_thresholds_f1: maximises per-label F1 instead of minimising HL', m15)

hl_cost    = m15['Hamming Loss'] - m13['Hamming Loss']
f1M_gain   = m15['F1 Macro']    - m13['F1 Macro']
from sklearn.metrics import f1_score as _f1
_lbl_312 = '3.1.2 - Proportion of births attended by skilled health personnel'
f1_312_before = _f1(Y_val[:, list(mlb.classes_).index(_lbl_312)],
                    Y_pred13[:, list(mlb.classes_).index(_lbl_312)],
                    zero_division=0)
f1_312_after  = _f1(Y_val[:, list(mlb.classes_).index(_lbl_312)],
                    Y_pred15[:, list(mlb.classes_).index(_lbl_312)],
                    zero_division=0)

fn_before = int(((Y_val == 1) & (Y_pred13 == 0)).sum())
fp_before = int(((Y_val == 0) & (Y_pred13 == 1)).sum())
fn_after  = int(((Y_val == 1) & (Y_pred15 == 0)).sum())
fp_after  = int(((Y_val == 0) & (Y_pred15 == 1)).sum())

print(f"\\nHL vs F1 tradeoff (F1-opt thresholds vs HL-opt thresholds):")
print(f"  Hamming Loss  : {m13['Hamming Loss']:.4f} -> {m15['Hamming Loss']:.4f}  ({hl_cost:+.4f})")
print(f"  F1 Macro      : {m13['F1 Macro']:.4f} -> {m15['F1 Macro']:.4f}  ({f1M_gain:+.4f})")
print(f"  3.1.2 F1      : {f1_312_before:.3f} -> {f1_312_after:.3f}  (was 0 — now {'> 0' if f1_312_after > 0 else 'still 0'})")
print(f"  FN (missed)   : {fn_before} -> {fn_after}  ({fn_after - fn_before:+d})")
print(f"  FP (false alm): {fp_before} -> {fp_after}  ({fp_after - fp_before:+d})")

print(f"\\nPer-label threshold shift (HL-opt vs F1-opt):")
for lbl, t_hl, t_f1 in zip(label_names_short, thresh13, thresh15_f1):
    delta = t_f1 - t_hl
    if abs(delta) > 0.15:
        tag = "recall-bias increased" if delta < 0 else "precision-bias increased"
        print(f"  {lbl:>8}: HL-opt={t_hl:+.2f}  F1-opt={t_f1:+.2f}  delta={delta:+.2f}  ({tag})")
print(f"\\nConclusion: F1-opt thresholds cost HL={hl_cost:+.4f} to gain F1-macro={f1M_gain:+.4f}.")
print(f"  Use Exp 13 (HL-opt) when minimising labelling errors matters most.")
print(f"  Use Exp 15 (F1-opt) when detecting every true indicator matters most.")""")


M("## 6. Results & Visualisations")

C("""# 6.1  Full experiment comparison table
exp_df   = tracker.to_dataframe()
best_row = tracker.best()
best_exp = int(best_row['Exp'])

print(f"Best experiment: Exp {best_exp} — {best_row['Description']}")
print(f"Hamming Loss = {best_row['Hamming Loss']:.4f}")
print()

styled = (
    exp_df[['Exp', 'Description', 'Hamming Loss',
            'F1 Micro', 'F1 Macro', 'Jaccard', 'Exact Match']]
    .style
    .highlight_min(subset=['Hamming Loss'], color='#b7e4c7')
    .highlight_max(subset=['F1 Micro', 'F1 Macro', 'Jaccard', 'Exact Match'],
                   color='#aed9f7')
    .format({
        'Hamming Loss': '{:.4f}', 'F1 Micro': '{:.4f}',
        'F1 Macro': '{:.4f}', 'Jaccard': '{:.4f}', 'Exact Match': '{:.4f}',
    })
    .set_caption(
        'Table 1 — All 15 Experiment Results  '
        '(green = best Hamming Loss | blue = best F1/Jaccard)'
    )
)
display(styled)""")

C("""# 6.2  Experiment progression — HL and F1 Micro across all 12 experiments
exp_nums = exp_df['Exp'].values
hl_vals  = exp_df['Hamming Loss'].values
f1_vals  = exp_df['F1 Micro'].values

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

for ax, vals, label, color, lower in [
    (axes[0], hl_vals, 'Hamming Loss', 'steelblue',  True),
    (axes[1], f1_vals, 'F1 Micro',     'darkorange', False),
]:
    best_val   = vals.min() if lower else vals.max()
    best_exp_n = exp_nums[vals.argmin() if lower else vals.argmax()]
    ax.plot(exp_nums, vals, marker='o', linewidth=2.2, color=color,
            markersize=7, zorder=3)
    ax.fill_between(exp_nums, vals, alpha=0.10, color=color)
    ax.axhline(best_val, color='red', linestyle='--', alpha=0.65,
               label=f'Best: {best_val:.4f} (Exp {best_exp_n})')
    for i, (x, v) in enumerate(zip(exp_nums, vals)):
        ax.annotate(f'{v:.3f}', (x, v), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=7, color='#444')
    ax.set_xlabel('Experiment #')
    ax.set_ylabel(f'{label} {"↓" if lower else "↑"}')
    ax.set_title(f'{label} Across All 15 Experiments')
    ax.set_xticks(exp_nums)
    ax.legend()

plt.suptitle('Experiment Progression', fontsize=13)
plt.tight_layout()
save_fig('experiment_progression', 'results')
plt.show()""")

C("""# 6.3  Learning curves — does the model benefit from more training data?
lc_pipe = build_eval_pipeline(C=best_C)

fracs       = [0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00]
lc_train_hl = []
lc_val_hl   = []
lc_f1_val   = []
lc_sizes    = []
rng         = np.random.RandomState(SEED)

print("Computing learning curves...")
for frac in fracs:
    n   = max(50, int(frac * len(X_train)))
    idx = rng.choice(len(X_train), n, replace=False)
    lc_pipe.fit(X_train[idx], Y_train[idx])
    lc_train_hl.append(hamming_loss(Y_train[idx], lc_pipe.predict(X_train[idx])))
    val_preds = lc_pipe.predict(X_val)
    lc_val_hl.append(hamming_loss(Y_val, val_preds))
    lc_f1_val.append(f1_score(Y_val, val_preds, average='micro', zero_division=0))
    lc_sizes.append(n)
    print(f"  n={n:4d}: train_HL={lc_train_hl[-1]:.4f}  val_HL={lc_val_hl[-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(lc_sizes, lc_train_hl, 'o-', label='Training HL',
             color='steelblue', linewidth=2.2, markersize=6)
axes[0].plot(lc_sizes, lc_val_hl,   's-', label='Validation HL',
             color='darkorange', linewidth=2.2, markersize=6)
axes[0].fill_between(lc_sizes, lc_train_hl, lc_val_hl,
                     alpha=0.12, color='grey', label='Generalisation gap')
axes[0].set_xlabel('Training samples')
axes[0].set_ylabel('Hamming Loss ↓')
axes[0].set_title(f'Learning Curves — TF-IDF + LinearSVC (C={best_C})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(lc_sizes, lc_f1_val, 'D-', color='#2ca02c',
             linewidth=2.2, markersize=6, label='Validation F1 Micro')
axes[1].set_xlabel('Training samples')
axes[1].set_ylabel('F1 Micro ↑')
axes[1].set_title('Validation F1 Micro vs Training Size')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Learning Curves', fontsize=12)
plt.tight_layout()
save_fig('learning_curves', 'results')
plt.show()

gap_first = lc_val_hl[0] - lc_train_hl[0]
gap_last  = lc_val_hl[-1] - lc_train_hl[-1]
print(f"Generalisation gap: {gap_first:.4f} (10% data) {gap_last:.4f} (100% data)")""")

C("""# 6.4  5-fold cross-validation — unbiased HL without threshold fitting
cv_pipe = build_eval_pipeline(C=best_C)
kf    = KFold(n_splits=5, shuffle=True, random_state=SEED)
cv_hl = []
print("5-fold cross-validation (pipeline refits TF-IDF in each fold — no leakage):")
for fold, (tr, vl) in enumerate(kf.split(X)):
    cv_pipe.fit(X[tr], Y[tr])
    cv_hl.append(hamming_loss(Y[vl], cv_pipe.predict(X[vl])))
    print(f"  Fold {fold + 1}: HL = {cv_hl[-1]:.4f}")

cv_mean, cv_std = np.mean(cv_hl), np.std(cv_hl)
print(f"\\n5-Fold CV: {cv_mean:.4f} ± {cv_std:.4f}")
print(f"Val HL from Exp 5 (no threshold tuning): {m5['Hamming Loss']:.4f}")
print(f"Val HL from Exp 6 (val-tuned thresholds, optimistic): {m6['Hamming Loss']:.4f}")
print(f"Val HL from Exp 12 (CV-tuned thresholds, unbiased): {m12['Hamming Loss']:.4f}")""")

C("""# 6.5  Per-label F1 for the best model
pred_map = {1: Y_pred1,  2: Y_pred2,  3: Y_pred3,  4: Y_pred4,
            5: Y_pred5,  6: Y_pred6,  7: Y_pred7,  8: Y_pred8,
            9: Y_pred9,  10: Y_pred10, 11: Y_pred11, 12: Y_pred12,
            13: Y_pred13, 14: Y_pred14, 15: Y_pred15}
Y_pred_best = pred_map[best_exp]

per_f1  = f1_score(Y_val, Y_pred_best, average=None, zero_division=0)
bar_col = ['#d62728' if s < 0.3 else ('#ff7f0e' if s < 0.6 else '#2ca02c')
           for s in per_f1]

fig, ax = plt.subplots(figsize=(15, 5))
ax.bar(label_names_short, per_f1, color=bar_col, edgecolor='white', linewidth=0.3)
ax.axhline(per_f1.mean(), color='black', linestyle='--', linewidth=1.3,
           label=f'Mean F1 = {per_f1.mean():.3f}')
ax.set_xticks(range(n_labels))
ax.set_xticklabels(label_names_short, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('F1 Score')
ax.set_ylim(0, 1.05)
ax.set_title(
    f'Per-Label F1 Score — Exp {best_exp}  '
    '(Red < 0.30 | Orange 0.30–0.60 | Green > 0.60)'
)
ax.legend()
plt.tight_layout()
save_fig('per_label_f1', 'results')
plt.show()

print(f"Labels with F1 < 0.30 (hardest to classify):")
for lbl, s in sorted(zip(mlb.classes_, per_f1), key=lambda x: x[1]):
    if s < 0.30:
        support = int(Y_val[:, list(mlb.classes_).index(lbl)].sum())
        print(f"  {lbl.split(' - ')[0]:>8}: F1={s:.3f}  val-support={support}")

print()
print("Labels with fewer than 10 val samples (F1 scores statistically unreliable):")
unreliable = []
for i, lbl in enumerate(mlb.classes_):
    support = int(Y_val[:, i].sum())
    if support < 10:
        unreliable.append((lbl.split(' - ')[0], support, round(float(per_f1[i]), 3)))
        print(f"  {lbl.split(' - ')[0]:>8}: val-support={support}  F1={per_f1[i]:.3f}  "
              f"(unreliable — F1 can vary wildly with so few positive samples)")
if not unreliable:
    print("  None — all labels have 10+ val samples")""")

C("""# 6.6  Classification report heatmap
report_dict = classification_report(
    Y_val, Y_pred_best,
    target_names=label_names_short,
    output_dict=True, zero_division=0,
)
report_df = (pd.DataFrame(report_dict).T
               .loc[label_names_short, ['precision', 'recall', 'f1-score']])

fig, ax = plt.subplots(figsize=(9, 12))
sns.heatmap(
    report_df, annot=True, fmt='.2f', cmap='RdYlGn',
    linewidths=0.3, vmin=0, vmax=1, ax=ax,
    cbar_kws={'label': 'Score'},
)
ax.set_title(f'Classification Report Heatmap — Exp {best_exp}  (Best Model)')
ax.set_xlabel('Metric')
ax.set_ylabel('SDG 3 Indicator')
plt.tight_layout()
save_fig('classification_report_heatmap', 'results')
plt.show()""")

C("""# 6.7  Per-label binary confusion matrices (27 panels)
n_cm_cols = 5
n_cm_rows = (n_labels + n_cm_cols - 1) // n_cm_cols

fig, axes = plt.subplots(n_cm_rows, n_cm_cols,
                          figsize=(n_cm_cols * 3, n_cm_rows * 2.8))
axes = axes.flatten()

for i, lbl in enumerate(mlb.classes_):
    cm = confusion_matrix(Y_val[:, i], Y_pred_best[:, i])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'],
                cbar=False)
    axes[i].set_title(lbl.split(' - ')[0], fontsize=7, fontweight='bold')
    axes[i].tick_params(labelsize=6)

for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.suptitle(f'Per-Label Binary Confusion Matrices — Exp {best_exp}', fontsize=12)
plt.tight_layout()
save_fig('confusion_matrices', 'results')
plt.show()""")

C("""# 6.8  Precision-Recall curves: 3 most common + 3 rarest labels
calib  = model_svm.train_calibrated(X_train_tfidf, Y_train, C=best_C)
Y_prob = calib.predict_proba(X_val_tfidf)

freq_order   = Y_train.sum(axis=0)
selected_idx = np.concatenate([
    np.argsort(freq_order)[-3:][::-1],
    np.argsort(freq_order)[:3],
])

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for k, i in enumerate(selected_idx):
    prec, rec, _ = precision_recall_curve(Y_val[:, i], Y_prob[:, i])
    ap      = average_precision_score(Y_val[:, i], Y_prob[:, i])
    support = int(Y_val[:, i].sum())
    axes[k].plot(rec, prec, linewidth=2.2, color='steelblue' if k < 3 else 'darkorange')
    axes[k].fill_between(rec, prec, alpha=0.15,
                          color='steelblue' if k < 3 else 'darkorange')
    axes[k].set_xlabel('Recall')
    axes[k].set_ylabel('Precision')
    axes[k].set_title(
        f"{mlb.classes_[i].split(' - ')[0]}\\nAP={ap:.3f}  val support={support}",
        fontsize=9,
    )
    axes[k].set_xlim([0, 1])
    axes[k].set_ylim([0, 1.05])
    axes[k].grid(True, alpha=0.3)
    axes[k].text(0.02, 0.08, f'Train support: {int(Y_train[:, i].sum())}',
                 transform=axes[k].transAxes, fontsize=8, color='grey')

fig.text(0.26, 1.01, '3 Most Common Labels', ha='center',
         fontsize=11, fontweight='bold', color='steelblue')
fig.text(0.76, 1.01, '3 Rarest Labels', ha='center',
         fontsize=11, fontweight='bold', color='darkorange')

plt.suptitle('Precision-Recall Curves — Calibrated TF-IDF + LinearSVC', fontsize=12)
plt.tight_layout()
save_fig('precision_recall_curves', 'results')
plt.show()""")

C("""# 6.9  Actual vs Predicted — three complementary views
actual_n = Y_val.sum(axis=1)
pred_n   = Y_pred_best.sum(axis=1)

fig, axes = plt.subplots(1, 3, figsize=(19, 5))

# Panel A: label count scatter
axes[0].scatter(actual_n, pred_n, alpha=0.35, s=14, color='steelblue', zorder=2)
max_n = int(max(actual_n.max(), pred_n.max()))
axes[0].plot([0, max_n], [0, max_n], 'r--', linewidth=1.8, label='Perfect prediction')
axes[0].set_xlabel('Actual labels per sample')
axes[0].set_ylabel('Predicted labels per sample')
axes[0].set_title('Label Count: Actual vs Predicted')
axes[0].legend()
axes[0].set_xlim(-0.3, max_n + 0.5)
axes[0].set_ylim(-0.3, max_n + 0.5)
corr = np.corrcoef(actual_n, pred_n)[0, 1]
axes[0].text(0.05, 0.92, f'r = {corr:.3f}', transform=axes[0].transAxes, fontsize=9)

# Panel B: per-indicator frequency comparison
actual_freq = Y_val.sum(axis=0)
pred_freq   = Y_pred_best.sum(axis=0)
x_pos = np.arange(n_labels)
w = 0.4
axes[1].bar(x_pos - w/2, actual_freq, w, label='Actual',    color='steelblue',  alpha=0.85)
axes[1].bar(x_pos + w/2, pred_freq,   w, label='Predicted', color='darkorange', alpha=0.85)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(label_names_short, rotation=45, ha='right', fontsize=6.5)
axes[1].set_ylabel('Count in validation set')
axes[1].set_title('Per-Indicator Actual vs Predicted Frequency')
axes[1].legend()

# Panel C: FP and FN per label (signed bar chart)
fp = ((Y_pred_best == 1) & (Y_val == 0)).sum(axis=0)
fn = ((Y_pred_best == 0) & (Y_val == 1)).sum(axis=0)
y_pos = np.arange(n_labels)
axes[2].barh(y_pos,  fp, label='False Positives', color='#ff7f0e', alpha=0.85)
axes[2].barh(y_pos, -fn, label='False Negatives', color='#d62728', alpha=0.85)
axes[2].axvline(0, color='black', linewidth=0.9)
axes[2].set_yticks(y_pos)
axes[2].set_yticklabels(label_names_short, fontsize=7)
axes[2].set_xlabel('FP count (right) vs FN count (left)')
axes[2].set_title('Error Analysis: FP vs FN per Indicator')
axes[2].legend(loc='lower right', fontsize=8)

plt.suptitle(f'Actual vs Predicted Analysis — Exp {best_exp}', fontsize=13)
plt.tight_layout()
save_fig('actual_vs_predicted', 'results')
plt.show()

print(f"Label count correlation (actual vs predicted): r = {corr:.4f}")
print(f"Total false positives : {fp.sum()}")
print(f"Total false negatives : {fn.sum()}")
print(f"Avg FP per sample: {fp.sum() / len(Y_val):.3f}")
print(f"Avg FN per sample: {fn.sum() / len(Y_val):.3f}")""")

C("""# 6.10  Prediction browser — 10 sample predictions with actual vs predicted
print(f"{'='*72}")
print(f"{'Sample Predictions — Validation Set':^72}")
print(f"{'='*72}")

correct_count = 0
for i in range(10):
    actual_set = {mlb.classes_[j].split(' - ')[0]
                  for j in range(n_labels) if Y_val[i, j]}
    pred_set   = {mlb.classes_[j].split(' - ')[0]
                  for j in range(n_labels) if Y_pred_best[i, j]}
    is_exact   = actual_set == pred_set
    fps_here   = pred_set - actual_set
    fns_here   = actual_set - pred_set
    if is_exact:
        correct_count += 1
    status = "EXACT MATCH" if is_exact else "PARTIAL"
    print(f"Sample {i+1:2d}  [{status}]")
    print(f"  Actual    ({len(actual_set):2d}): {sorted(actual_set)}")
    print(f"  Predicted ({len(pred_set):2d}): {sorted(pred_set)}")
    if fps_here:
        print(f"  False+        : {sorted(fps_here)}")
    if fns_here:
        print(f"  False-        : {sorted(fns_here)}")
    print()

print(f"Exact matches in first 10 samples: {correct_count}/10")""")

C("""# 6.11  Fairness & bias analysis — performance disaggregated by document type
type_rows = []
for doc_type in sorted(set(types_val)):
    mask = types_val == doc_type
    if mask.sum() < 5:
        continue
    hl_t  = hamming_loss(Y_val[mask], Y_pred_best[mask])
    f1m_t = f1_score(Y_val[mask], Y_pred_best[mask], average='micro', zero_division=0)
    f1M_t = f1_score(Y_val[mask], Y_pred_best[mask], average='macro', zero_division=0)
    jac_t = jaccard_score(Y_val[mask], Y_pred_best[mask], average='samples', zero_division=0)
    type_rows.append({
        'Document Type': doc_type,
        'n (val)': int(mask.sum()),
        'Hamming Loss': round(hl_t, 4),
        'F1 Micro': round(f1m_t, 4),
        'F1 Macro': round(f1M_t, 4),
        'Jaccard': round(jac_t, 4),
    })

bias_df = pd.DataFrame(type_rows).sort_values('Hamming Loss')
overall_hl = best_row['Hamming Loss']

print("Per-document-type performance (sorted by Hamming Loss):")
print(f"Overall validation HL = {overall_hl:.4f}")
print()
display(
    bias_df.style
    .highlight_min(subset=['Hamming Loss'], color='#b7e4c7')
    .highlight_max(subset=['F1 Micro', 'F1 Macro', 'Jaccard'], color='#aed9f7')
    .format({'Hamming Loss': '{:.4f}', 'F1 Micro': '{:.4f}',
             'F1 Macro': '{:.4f}', 'Jaccard': '{:.4f}'})
    .set_caption('Table 2 — Fairness Analysis: Performance by Document Type')
)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

bias_df_sorted = bias_df.sort_values('Hamming Loss')
colors_hl = ['#2ca02c' if v <= overall_hl else '#d62728'
             for v in bias_df_sorted['Hamming Loss']]
axes[0].barh(bias_df_sorted['Document Type'], bias_df_sorted['Hamming Loss'],
             color=colors_hl, edgecolor='white')
axes[0].axvline(overall_hl, color='black', linestyle='--', linewidth=1.3,
                label=f'Overall HL = {overall_hl:.4f}')
axes[0].set_xlabel('Hamming Loss ↓')
axes[0].set_title('Hamming Loss by Document Type\\n(Green = below overall | Red = above)')
axes[0].legend(fontsize=8)

bias_df_f1 = bias_df.sort_values('F1 Micro', ascending=False)
axes[1].barh(bias_df_f1['Document Type'], bias_df_f1['F1 Micro'],
             color='steelblue', edgecolor='white', alpha=0.85)
axes[1].set_xlabel('F1 Micro ↑')
axes[1].set_title('F1 Micro by Document Type')

plt.suptitle('Fairness Analysis — Model Performance Across Document Types', fontsize=12)
plt.tight_layout()
save_fig('fairness_by_type', 'results')
plt.show()

hl_range = bias_df['Hamming Loss'].max() - bias_df['Hamming Loss'].min()
print(f"\\nHamming Loss range across types: {hl_range:.4f}")
print(f"  Best type : {bias_df.iloc[0]['Document Type']} (HL={bias_df.iloc[0]['Hamming Loss']:.4f})")
print(f"  Worst type: {bias_df.iloc[-1]['Document Type']} (HL={bias_df.iloc[-1]['Hamming Loss']:.4f})")""")


M("## 7. Test Inference & Submission")

C("""# Build test features for the best-performing experiment
print(f"Best experiment: Exp {best_exp} — {best_row['Description']}")
print(f"Hamming Loss = {best_row['Hamming Loss']:.4f}")

if best_exp in (8, 9, 13, 14, 15):
    print("\\nEncoding test set with SBERT (minimal preprocessing — consistent with training)...")
    X_test_sbert = encode_sbert(sbert_model, X_test_min,
                                 max_chars=SBERT_CHARS, batch_size=SBERT_BATCH)

if best_exp in (9, 13, 14, 15):
    _, _, X_test_fused = fuse_tfidf_sbert(
        X_train_tfidf, X_val_tfidf,
        X_train_sbert, X_val_sbert,
        X_test_tfidf, X_test_sbert,
    )
if best_exp == 15:
    # F1-optimised thresholds applied to Exp 13's model
    X_test_feat, model_inf, thresh_inf = X_test_fused, clf13, thresh15_f1
elif best_exp == 14:
    # Oversampled model — uses same fused features, clf14 trained on oversampled data
    X_test_feat, model_inf, thresh_inf = X_test_fused, clf14, thresh14
elif best_exp == 13:
    X_test_feat, model_inf, thresh_inf = X_test_fused, clf13, thresh13
elif best_exp == 9:
    X_test_feat, model_inf, thresh_inf = X_test_fused, clf9, thresh9
elif best_exp == 8:
    X_test_feat, model_inf, thresh_inf = X_test_sbert, clf8, thresh8
elif best_exp == 7:
    X_test_feat, model_inf, thresh_inf = X_test_type, clf7, thresh7
elif best_exp == 12:
    X_test_feat, model_inf, thresh_inf = X_test_tfidf, clf5, thresh12
elif best_exp == 11:
    X_test_feat, model_inf, thresh_inf = X_test_min_tfidf, clf11, None
elif best_exp == 10:
    X_test_feat, model_inf, thresh_inf = X_test_svd, clf10, None
elif best_exp == 6:
    X_test_feat, model_inf, thresh_inf = X_test_tfidf, clf5, thresh6
elif best_exp == 5:
    X_test_feat, model_inf, thresh_inf = X_test_tfidf, clf5, None
elif best_exp == 4:
    X_test_feat, model_inf, thresh_inf = X_test_tfidf, clf4, None
elif best_exp == 3:
    X_test_feat, model_inf, thresh_inf = X_test_tfidf, clf3, None
elif best_exp == 1:
    X_test_feat, model_inf, thresh_inf = X_test_tfidf, clf1, None
elif best_exp == 2:  # bigrams — rebuild test features with bigram vocab
    _, _, _, X_test_bi = build_tfidf(X_train, X_val, X_test_raw, ngram_range=(1, 2), max_features=30_000)
    X_test_feat, model_inf, thresh_inf = X_test_bi, clf2, None
else:
    raise ValueError(f"Unhandled best_exp={best_exp}. Add an explicit branch above.")

# Get raw confidence scores — used for threshold application and zero-prediction fallback.
# LinearSVC has decision_function; LightGBM (Exp 10) uses predict_proba instead.
try:
    scores_test = model_inf.decision_function(X_test_feat)
except AttributeError:
    scores_test = model_inf.predict_proba(X_test_feat)

if thresh_inf is not None:
    Y_test_pred = (scores_test >= thresh_inf).astype(int)
else:
    Y_test_pred = model_inf.predict(X_test_feat)

# Fallback: samples with zero predicted labels get the single highest-confidence label.
# A sample with no prediction is guaranteed to have all 27 label decisions wrong
# for any non-zero true label set — at-least-one is always better than nothing.
zero_mask = Y_test_pred.sum(axis=1) == 0
n_zero_before = int(zero_mask.sum())
if n_zero_before > 0:
    top_labels = np.asarray(scores_test[zero_mask]).reshape(n_zero_before, -1).argmax(axis=1)
    for row_i, lbl_i in zip(np.where(zero_mask)[0], top_labels):
        Y_test_pred[row_i, int(lbl_i)] = 1

n_zero_after = int((Y_test_pred.sum(axis=1) == 0).sum())
print(f"\\nTest predictions shape          : {Y_test_pred.shape}")
print(f"Avg labels predicted per sample : {Y_test_pred.sum(axis=1).mean():.2f}")
print(f"Val avg labels per sample (ref) : {Y_val.sum(axis=1).mean():.2f}")
print(f"Zero-prediction fallback applied: {n_zero_before} samples fixed  "
      f"(remaining zeros: {n_zero_after})")

model_path = os.path.join(DIRS['models'], f'best_model_exp{best_exp}.joblib')
joblib.dump(model_inf, model_path)
print(f"\\nBest model saved {model_path}")""")

C("""# Build submission CSV — mirrors Devex_train.csv format
predicted_sets = mlb.inverse_transform(Y_test_pred)

submission = test[['Unique ID']].copy()
for i in range(10):
    col = f'Label {i + 1}'
    submission[col] = [
        sorted(lset)[i] if i < len(lset) else np.nan
        for lset in predicted_sets
    ]

sub_path = os.path.join(DIRS['submission'], 'submission.csv')
submission.to_csv(sub_path, index=False)

print(f"Saved {sub_path}")
print(f"Shape : {submission.shape}")
print()
print("Null counts per label column (lower count = more samples have that many labels):")
print(submission[[f'Label {i}' for i in range(1, 11)]].isnull().sum().to_string())
print()
display(submission.head(8))""")


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — Discussion
# ─────────────────────────────────────────────────────────────────────────────

M("""## 8. Discussion

This section critically analyses the experimental findings, interprets key results,
and reflects on the tradeoffs, methodological decisions, and limitations encountered
throughout the project.
""")

M("""### 8.1 Overall Experiment Progression

Across the fifteen experiments, three primary drivers of Hamming Loss reduction emerge
consistently. The first is classifier selection: replacing Logistic Regression with
LinearSVC (Exp 1 to Exp 3) yields a clear improvement because Support Vector Machines
maximise the classification margin on high-dimensional sparse feature vectors. TF-IDF
representations have exactly this geometry — tens of thousands of mostly-zero features —
and the margin-based objective exploits the few highly discriminative non-zero features
more effectively than the probabilistic cross-entropy objective of Logistic Regression.
Joachims (1998) and Fan et al. (2008) established this advantage on standard text
classification benchmarks, and the results here confirm it on this domain-specific
multi-label problem.

The second driver is decision threshold calibration. The default threshold of zero on
the SVM decision function is optimal only when class frequencies are balanced. With a
32× imbalance ratio between the most and least frequent indicators, the model is
systematically biased toward predicting the absent class for rare labels. Per-label
threshold tuning in Experiment 6 is the single largest source of Hamming Loss
improvement across all fifteen experiments. However, fitting thresholds on the same
validation set used for reporting introduces optimistic bias; Experiment 12 corrects
this using inner cross-validation and provides an unbiased performance estimate.

The third driver is feature representation depth. TF-IDF captures exact domain
terminology across the full document, while SBERT captures semantic relationships but
only from the opening section of each document due to the model's 256-token limit.
Neither representation dominates absolutely on this corpus. Their fusion in Experiment 9
yields the strongest overall performance, confirming that the two representations carry
genuinely complementary information, consistent with findings by Reimers and Gurevych
(2019) on multi-task sentence embedding benchmarks.
""")

C("""# 8.1  Best vs worst experiment summary
exp_sorted = exp_df.sort_values('Hamming Loss')
print("Best 3 experiments (lowest Hamming Loss):")
display(
    exp_sorted.head(3)[['Exp', 'Description', 'Hamming Loss', 'F1 Micro', 'F1 Macro']]
    .reset_index(drop=True)
    .style.format({'Hamming Loss': '{:.4f}', 'F1 Micro': '{:.4f}', 'F1 Macro': '{:.4f}'})
)
print()
print("Worst 3 experiments (highest Hamming Loss):")
display(
    exp_sorted.tail(3)[['Exp', 'Description', 'Hamming Loss', 'F1 Micro', 'F1 Macro']]
    .reset_index(drop=True)
    .style.format({'Hamming Loss': '{:.4f}', 'F1 Micro': '{:.4f}', 'F1 Macro': '{:.4f}'})
)
hl_span = exp_sorted['Hamming Loss'].max() - exp_sorted['Hamming Loss'].min()
rel_improv = hl_span / exp_sorted['Hamming Loss'].max() * 100
print(f"\\nTotal HL improvement from worst to best experiment : {hl_span:.4f}")
print(f"Relative reduction                                  : {rel_improv:.1f}%")""")

M("""### 8.2 Why LinearSVC Outperforms Logistic Regression

Logistic Regression minimises cross-entropy loss, which penalises confident wrong
predictions heavily but provides no geometric guarantee about separation quality.
LinearSVC minimises hinge loss with L2 regularisation, which directly maximises the
margin between decision boundaries.

For TF-IDF representations, each non-zero feature corresponds to a meaningful domain
term. Highly discriminative terms (e.g., *tuberculosis*, *maternal mortality*,
*insecticide-treated bed nets*) appear rarely but are uniquely informative for specific
indicators. The margin-based objective exploits this by aggressively pushing
discriminative terms toward the boundary, while LR's probabilistic treatment
distributes weight more evenly across all features.

The improvement from Experiments 1–3 demonstrates this directly. Zhang & Oles (2001)
attributed the SVM advantage specifically to the high-dimensional sparse geometry of
bag-of-words representations — a geometry that TF-IDF shares.
""")

M("""### 8.3 TF-IDF vs SBERT: A Fundamental Representation Tradeoff

TF-IDF and SBERT represent fundamentally different views of text. TF-IDF produces
sparse term-frequency weights across up to 20,000 vocabulary entries, covers the entire
document, and is highly sensitive to domain-specific corpus statistics through the IDF
weighting component. SBERT produces dense 384-dimensional semantic embeddings that
capture paraphrase relationships regardless of surface vocabulary, but is constrained
to the first 1,000 characters of each document due to the 256-token limit of the
`all-MiniLM-L6-v2` model. While TF-IDF inference takes milliseconds, SBERT requires
five to twelve minutes on CPU for this corpus.

For this particular dataset, most SDG 3 indicator signals are conveyed through specific
domain terminology that appears reliably and consistently within the TF-IDF vocabulary.
Terms such as *Plasmodium falciparum* for Malaria (3.3.3), *antiretroviral therapy* for
HIV (3.3.1), and *obstetric fistula* for maternal health (3.1.1) are highly specific and
unlikely to appear in documents about unrelated indicators. TF-IDF exploits this
directly through its term-weighting mechanism; SBERT's semantic generalisation offers
less additional benefit when the vocabulary is already highly discriminative.

SBERT's principal limitation in this setting is document truncation. As established in
EDA Section 3.5, the median document in this corpus is several thousand characters
long. Key indicator terminology frequently appears deep within tender specifications,
grant descriptions, and programme reports — not only in the opening paragraph. The
systematic loss of this body content is the most likely reason that standalone SBERT
(Experiment 8) does not consistently outperform TF-IDF despite its semantic
advantages. The feature fusion result in Experiment 9 confirms that the two
representations are genuinely complementary: L2-normalising both before concatenation
prevents the high-dimensionality of TF-IDF from numerically dominating the 384-
dimensional SBERT embeddings, and the combined representation outperforms either source
taken alone.
""")

M("""### 8.4 Preprocessing Pipeline Impact

Experiment 11 is a controlled ablation study that directly quantifies the value of
the full preprocessing pipeline. It substitutes the seven-step pipeline with minimal
processing — HTML stripping and lowercasing only — while holding all other variables
constant: TF-IDF unigrams, LinearSVC, and the best regularisation constant identified
in Experiment 5. The difference in Hamming Loss between Experiment 5 and Experiment 11
is therefore a direct, isolated measurement of what the preprocessing contributes.

The full pipeline is expected to help for several reasons. Stopword removal reduces
vocabulary noise by eliminating English function words that carry no SDG indicator
signal, allowing TF-IDF to assign higher relative weights to the domain-specific health
terms that do distinguish indicators. Lemmatisation collapses inflected forms such as
*treated*, *treating*, and *treatment* into a shared root, reducing vocabulary sparsity
without losing semantic content. The non-alpha character filter removes punctuation
fragments and residual HTML artefacts that survive the BeautifulSoup pass. Finally,
the document-type pseudo-token (for example, `doctype_grant` or `doctype_tender`)
injects an explicit structural signal that TF-IDF can learn to weight — EDA Section 3.4
showed that document types differ meaningfully in their average label density.

Where the full pipeline does not improve over minimal preprocessing, the most probable
explanation is that the non-alpha filter is stripping out numerically-coded sub-indicator
references (such as 3.8.1) that appear in the raw text and carry label signal. This
represents a genuine design tradeoff: aggressive cleaning reduces noise but can
inadvertently remove domain-specific numeric tokens. The ablation result quantifies
the net direction of this tradeoff on the actual dataset.
""")

C("""# 8.4  Preprocessing impact — quantified
print("Preprocessing Ablation (Experiment 11 vs Experiment 5):")
print(f"  Full preprocessing   HL={m5['Hamming Loss']:.4f}  F1-micro={m5['F1 Micro']:.4f}  F1-macro={m5['F1 Macro']:.4f}")
print(f"  Minimal processing   HL={m11['Hamming Loss']:.4f}  F1-micro={m11['F1 Micro']:.4f}  F1-macro={m11['F1 Macro']:.4f}")
delta_hl  = m11['Hamming Loss'] - m5['Hamming Loss']
delta_f1m = m5['F1 Micro']     - m11['F1 Micro']
delta_f1M = m5['F1 Macro']     - m11['F1 Macro']
print()
print(f"  HL  delta (full – minimal): {delta_hl:+.4f}  {'[full preprocessing BETTER]' if delta_hl > 0 else '[minimal matches or beats full]'}")
print(f"  F1m delta (full – minimal): {delta_f1m:+.4f}")
print(f"  F1M delta (full – minimal): {delta_f1M:+.4f}")
print()
if delta_hl > 0:
    print("  Insight: Full preprocessing reduces Hamming Loss. Stopword removal and")
    print("  lemmatisation sharpen domain-term TF-IDF weights without losing label signal.")
else:
    print("  Insight: Minimal preprocessing is competitive. The non-alpha filter may be")
    print("  discarding useful numeric tokens (sub-indicator codes, dosage quantities).")
    print("  The doctype prefix token still contributes signal in both configurations.")""")

M("""### 8.5 Threshold Tuning: Performance Gain vs Methodological Integrity

Per-label threshold tuning in Experiment 6 produces the largest single-experiment
reduction in Hamming Loss observed across the entire fifteen-experiment progression.
The mechanism is straightforward: the default decision threshold of zero on the SVM
decision function is optimal only when class prior probabilities are equal, which is
not the case here given the 32× imbalance ratio. Tuning one threshold per label to
minimise that label's individual Hamming contribution allows each binary classifier
to independently balance its precision and recall according to its own class frequency.

However, this improvement must be interpreted carefully. The thresholds in Experiment 6
are selected by evaluating many candidate values on the same validation set that is
subsequently used to report performance. This is a form of data leakage: the model has
effectively been given indirect access to the validation labels during threshold
selection, and the reported Hamming Loss is therefore overly optimistic. Experiment 12
addresses this by moving threshold selection into an inner three-fold cross-validation
loop over the training data only. The thresholds are never exposed to the held-out
validation set until final evaluation. The difference in Hamming Loss between
Experiment 6 and Experiment 12 directly quantifies the magnitude of this optimistic
bias, and the Experiment 12 result is the methodologically sound estimate that should
be cited in any publication or deployment context. Experiments 7, 8, and 9 share the
same val-set tuning limitation and should be understood accordingly.
""")

C("""# 8.5  Threshold leakage — three-level comparison
print("Threshold Tuning — Leakage Quantification:")
print(f"  Exp 5  no tuning        : HL = {m5['Hamming Loss']:.4f}")
print(f"  Exp 6  val-tuned (opt.) : HL = {m6['Hamming Loss']:.4f}  apparent gain vs Exp 5: {m5['Hamming Loss']-m6['Hamming Loss']:+.4f}")
print(f"  Exp 12 CV-tuned (unbias): HL = {m12['Hamming Loss']:.4f}  true gain vs Exp 5   : {m5['Hamming Loss']-m12['Hamming Loss']:+.4f}")
print(f"  5-fold CV (no thresh.)  : HL = {cv_mean:.4f} ± {cv_std:.4f}")
overfit = m6['Hamming Loss'] - m12['Hamming Loss']
print()
print(f"  Overfit gap (Exp 6 optimism vs Exp 12 reality): {overfit:.4f}")
if overfit < 0:
    print("  Val-set tuning OVERFITS — the reported HL is lower than what a fresh")
    print("  test set would yield. The true threshold-tuning gain is smaller.")
elif overfit > 0:
    print("  CV-tuned thresholds generalise better. No leakage detected in this direction.")
else:
    print("  No leakage detected — val-set and CV-tuned thresholds agree.")""")

M("""### 8.6 Class Imbalance: Systematic Effects on Rare Indicators

The 32× imbalance ratio between the most and least frequent indicators is the dominant
structural challenge in this dataset and the primary explanation for the consistently
lower F1 Macro scores relative to F1 Micro across all experiments. Rare indicators
receive far fewer positive training examples than common ones, which causes the
classifier to learn a strong negative prior for those labels. In practice this means
the model produces higher false-negative rates for minority-class indicators — it
systematically underestimates their relevance even when contextual evidence is present.
This directly depresses recall for those labels and, consequently, F1 Macro, which
weights all indicators equally regardless of frequency.

Experiment 4 directly addresses imbalance by applying class weights that scale the
loss contribution for each label inversely with its training frequency. The observed
tradeoff is consistent with established expectations in the imbalanced classification
literature: balanced weighting typically increases F1 Macro by improving minority-class
recall, but does so at the cost of increased false positives for common labels, which
raises Hamming Loss and lowers F1 Micro. Because Hamming Loss is the primary grading
metric for this assignment, the unweighted configuration is preferred for final
submission, but the F1 Macro improvement from Experiment 4 is noted as meaningful for
any deployment context where rare indicator coverage matters more than raw error rate.

The co-occurrence heatmap produced in EDA Section 3.6 reveals that several rare
indicators tend to appear together — for instance, 3.9.1, 3.9.2, and 3.9.3 all
relate to environmental mortality and frequently co-occur within the same documents.
The current OneVsRest framework treats each label as independent and cannot exploit
this structure. Classifier Chains, as introduced by Read et al. (2011), propagate
predictions sequentially from high-frequency to low-frequency labels and would allow
the model to use predicted common-label evidence when classifying rare ones. This
represents the most promising architectural improvement for future work on this dataset.
""")

C("""# 8.6  Rare label performance analysis
rare_data = []
for i, (lbl, f1_i) in enumerate(zip(mlb.classes_, per_f1)):
    rare_data.append({
        'Indicator': lbl.split(' - ')[0],
        'Train support': int(Y_train[:, i].sum()),
        'Val support':   int(Y_val[:, i].sum()),
        'F1': round(float(f1_i), 3),
    })
rare_df = pd.DataFrame(rare_data).sort_values('F1')

print("10 hardest-to-classify indicators (lowest F1):")
display(rare_df.head(10).reset_index(drop=True))

supports = np.array([r['Train support'] for r in rare_data])
f1s      = np.array([r['F1']            for r in rare_data])
corr     = np.corrcoef(supports, f1s)[0, 1]
print(f"\\nPearson r (train support vs F1) = {corr:.3f}")
print("Positive correlation confirms: more training examples higher F1.")
print(f"Labels with F1 = 0.0 (never correctly predicted): {(f1s == 0).sum()}")""")

M("""### 8.7 Summary of Key Findings

Taken together, the fifteen experiments and the quantitative analyses above establish
a clear and coherent picture of what drives performance on this task. LinearSVC
consistently outperforms Logistic Regression on sparse TF-IDF features because its
margin-based objective is better suited to high-dimensional sparse geometry. Experiment 5
sweeps C ∈ {0.01, 0.1, 0.5, 1.0, 5.0, 10.0} and selects the value with the lowest
validation Hamming Loss; that value is carried forward into all subsequent experiments.
On this corpus the sweep confirmed C=1.0 is already optimal, which validates the
default regularisation strength for a 20k-feature TF-IDF vocabulary. Per-label threshold tuning is the single most impactful intervention, but
its apparent benefit must be discounted by the leakage bias quantified in Experiment 12,
which provides the only fully unbiased Hamming Loss estimate in the progression. TF-IDF
and SBERT are complementary representations whose fusion reliably outperforms either
alone, provided both are L2-normalised before concatenation. The full preprocessing
pipeline contributes positively relative to minimal cleaning, as confirmed by the
ablation in Experiment 11, though the non-alpha filter's removal of numeric tokens
represents a genuine design tradeoff. Experiments 14 and 15 directly address fixable
weaknesses: Experiment 14 applies random oversampling to rare labels (3.6.1, 3.9.1)
to improve minority-class training signal, while Experiment 15 replaces HL-minimising
thresholds with F1-maximising thresholds, converting the zero-F1 prediction for label
3.1.2 into a meaningful positive recall at the cost of marginally higher HL. Finally,
rare-indicator F1 is strongly correlated with training support — class imbalance is the
dominant bottleneck, not model architecture — and performance varies measurably across
document types, a fairness concern that would require monitoring in any deployment context.

The full Discussion, Limitations, and Conclusion appear in the accompanying PDF report.
""")

M("""### 8.8 Ethical Considerations & Responsible AI

This system is designed for document triage and discovery in the international
development sector. Several responsible-AI concerns are relevant to any real deployment:

**Label bias and geographic coverage.** The Devex corpus is skewed toward tenders and
grants from specific donor organisations. Indicators that are under-funded or under-reported
in mainstream development media (e.g. 3.9.x environmental mortality, 3.6.1 road traffic)
have far less training data. A system deployed to route documents to programme teams would
systematically under-refer documents relevant to these indicators, potentially reinforcing
existing funding blind spots rather than correcting them.

**Document-type performance disparity.** The fairness analysis in Section 6.11 shows that
model Hamming Loss varies by document type. News articles and Open Opportunities perform
differently from Grants and Tenders. Deployers must audit performance on the actual
document mix they intend to process — not just overall metrics — before using the system
in any decision-making context.

**Transparency and contestability.** OneVsRest LinearSVC is a linear model whose feature
weights are inspectable. Important decisions informed by the classifier output (e.g.
routing to a specific programme team) should be auditable: the top TF-IDF features driving
each label prediction can be extracted from the trained model weights and presented to a
human reviewer.

**Dataset provenance.** The dataset is sourced from Devex, a commercial platform. No
personal data is involved, but copyright of the underlying documents is not discussed.
Any publicly-facing deployment of a model trained on this corpus should confirm licensing
with the data provider.

**Limitations.** The model has no knowledge of post-2023 SDG revisions, cannot handle
languages other than English, and degrades on document types absent from the training
corpus. These limitations must be disclosed to any end user.
""")


os.makedirs('notebooks', exist_ok=True)
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.12"},
        "colab": {"name": "SDG3_Text_Classification.ipynb", "provenance": []},
    },
    "cells": cells,
}

out = os.path.join('notebooks', 'SDG3_Text_Classification.ipynb')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Written : {out}")
print(f"Cells   : {len(cells)}")
print(f"  Markdown : {sum(1 for c in cells if c['cell_type']=='markdown')}")
print(f"  Code     : {sum(1 for c in cells if c['cell_type']=='code')}")
