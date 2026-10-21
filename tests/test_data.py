"""Unit tests for data loading and cleaning functions."""

import pytest
from newsclf.data import ID2LABEL, LABEL2ID, LABEL_NAMES, clean_text


def test_label_mappings():
    """Verify label mappings and integrity."""
    assert len(LABEL_NAMES) == 4
    assert LABEL_NAMES == ["World", "Sports", "Business", "Sci/Tech"]
    for i, name in enumerate(LABEL_NAMES):
        assert ID2LABEL[i] == name
        assert LABEL2ID[name] == i


def test_clean_text_html_stripping():
    """Verify HTML tags and entities are properly removed."""
    raw = "<b>Breaking News:</b> Stock markets surge &amp; rally! <a href='https://example.com'>Read more</a>"
    cleaned = clean_text(raw, lowercase=True)
    assert "<b>" not in cleaned
    assert "</b>" not in cleaned
    assert "<a" not in cleaned
    assert "&amp;" not in cleaned
    assert "breaking news: stock markets surge & rally! read more" in cleaned


def test_clean_text_whitespace():
    """Verify whitespace normalization."""
    raw = "   Headline \t\t with   unusual\n\nspacing   "
    cleaned = clean_text(raw, lowercase=True)
    assert cleaned == "headline with unusual spacing"


def test_clean_text_lowercase_flag():
    """Verify case preservation when lowercase=False (for Transformer models)."""
    raw = "Apple Launches New iPhone in California"
    cleaned_lower = clean_text(raw, lowercase=True)
    cleaned_preserve = clean_text(raw, lowercase=False)

    assert cleaned_lower == "apple launches new iphone in california"
    assert cleaned_preserve == "Apple Launches New iPhone in California"


def test_clean_text_edge_cases():
    """Verify edge cases like empty string, whitespace only, and None."""
    assert clean_text("") == ""
    assert clean_text("   ") == ""
    assert clean_text(None) == ""
