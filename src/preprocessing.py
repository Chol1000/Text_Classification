"""
Text preprocessing pipeline for SDG 3 indicator classification.

Steps applied in order:
  1. Strip HTML tags (BeautifulSoup)
  2. Lowercase
  3. Remove URLs
  4. Remove non-alphabetic characters
  5. Tokenise, filter stopwords and tokens shorter than 3 chars
  6. Lemmatise (WordNetLemmatizer)
  7. Optionally prepend a document-type pseudo-token  (e.g. doctype_grant)
"""

import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)

_lemmatizer = WordNetLemmatizer()
_stop_words  = set(stopwords.words('english'))


def preprocess(text, doc_type=None):
    """Clean a single document string.

    Parameters
    ----------
    text : str
        Raw document text (may contain HTML).
    doc_type : str or None
        Document type label (e.g. 'Grant', 'Tender').  When provided, a
        pseudo-token ``doctype_<type>`` is prepended to the token list.

    Returns
    -------
    str
        Space-joined cleaned token string.
    """
    text = BeautifulSoup(str(text), 'html.parser').get_text(separator=' ')
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [t for t in text.split() if t not in _stop_words and len(t) > 2]
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    if doc_type:
        tokens = [f'doctype_{doc_type.lower().replace(" ", "_")}'] + tokens
    return ' '.join(tokens)


def preprocess_minimal(text):
    """HTML strip and lowercase only — no stopwords, lemmatisation, or type token.

    Used for the ablation in Experiment 11 to isolate the contribution of the
    full preprocessing pipeline.
    """
    text = BeautifulSoup(str(text), 'html.parser').get_text(separator=' ')
    return text.lower().strip()


def preprocess_dataframe(df, text_col='Text', type_col='Type'):
    """Apply :func:`preprocess` to every row of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing text and type columns.
    text_col : str
        Name of the raw text column.
    type_col : str
        Name of the document-type column.

    Returns
    -------
    pd.Series
        Cleaned text, one entry per row.
    """
    return df.apply(lambda r: preprocess(r[text_col], r.get(type_col)), axis=1)
