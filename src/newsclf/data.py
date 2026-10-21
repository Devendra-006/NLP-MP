"""Data loading, cleaning, and splitting utilities for News Category Classifier."""

import html
import os
import re
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split

LABEL_NAMES: List[str] = ["World", "Sports", "Business", "Sci/Tech"]
ID2LABEL: Dict[int, str] = {i: name for i, name in enumerate(LABEL_NAMES)}
LABEL2ID: Dict[str, int] = {name: i for i, name in enumerate(LABEL_NAMES)}


# Precompiled regex patterns for high-throughput string cleaning
_RE_HTML = re.compile(r"<[^>]+>")
_RE_BACKSLASH = re.compile(r"\\+")
_RE_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str, lowercase: bool = True) -> str:
    """Perform basic text cleaning on news headlines or articles.

    Args:
        text: Raw input text string.
        lowercase: Whether to convert text to lowercase (True for TF-IDF, False for Transformer).

    Returns:
        Cleaned text string with HTML tags removed and normalized whitespace.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    # Fast path for empty strings
    if not text:
        return ""

    # Unescape HTML entities like &amp;, &lt;, etc.
    text = html.unescape(text)

    # Strip HTML tags
    text = _RE_HTML.sub(" ", text)

    # Replace backslash-escaped characters or common artifacts
    text = _RE_BACKSLASH.sub(" ", text)

    # Normalize whitespace (tabs, newlines, multiple spaces)
    text = _RE_WHITESPACE.sub(" ", text).strip()

    if lowercase:
        text = text.lower()

    return text


def load_raw_dataset(cache_dir: Optional[str] = "data") -> DatasetDict:
    """Load the AG News dataset using Hugging Face datasets library.

    Tries standard 'ag_news' first, with fallback to 'fancyzhx/ag_news'.
    """
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    try:
        ds = load_dataset("ag_news", cache_dir=cache_dir)
    except Exception:
        ds = load_dataset("fancyzhx/ag_news", cache_dir=cache_dir)

    return ds


def prepare_datasets(
    val_size: float = 0.1,
    random_state: int = 42,
    lowercase: bool = True,
    cache_dir: Optional[str] = "data",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load AG News, clean the text, and perform stratified train/validation split.

    Args:
        val_size: Fraction of train data to allocate to validation split (default: 0.1).
        random_state: Seed for stratified split reproducibility (default: 42).
        lowercase: Whether to lowercase during cleaning (True for TF-IDF, False for Transformer).
        cache_dir: Directory to cache datasets.

    Returns:
        train_df, val_df, test_df DataFrames with 'text' and 'label' columns.
    """
    ds = load_raw_dataset(cache_dir=cache_dir)

    train_raw = ds["train"].to_pandas()
    test_raw = ds["test"].to_pandas()

    # Drop any null or empty rows if present
    train_raw = train_raw.dropna(subset=["text", "label"])
    test_raw = test_raw.dropna(subset=["text", "label"])

    # Clean text
    train_raw["text"] = train_raw["text"].apply(lambda t: clean_text(t, lowercase=lowercase))
    test_raw["text"] = test_raw["text"].apply(lambda t: clean_text(t, lowercase=lowercase))

    # Stratified validation split on train set (10% of train, fixed seed 42)
    train_df, val_df = train_test_split(
        train_raw,
        test_size=val_size,
        stratify=train_raw["label"],
        random_state=random_state,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_raw.reset_index(drop=True)

    return train_df, val_df, test_df
