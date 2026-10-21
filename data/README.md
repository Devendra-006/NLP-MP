# 📁 AG News Dataset & Data Directory Guide

This directory stores cached dataset splits, benchmark sets, and out-of-domain evaluation data.

---

## 1. Primary Benchmark Dataset: AG News

The **AG News** dataset is a standard text classification benchmark created by Xiang Zhang from 496,835 news articles across more than 2,000 news sources.

### Official Benchmark Specifications
- **Classes**: 4 categories (`World`, `Sports`, `Business`, `Sci/Tech`)
- **Total Samples**: 127,600 samples
  - **Train Set**: 120,000 samples (30,000 balanced samples per class)
  - **Test Set**: 7,600 samples (1,900 balanced samples per class)
- **Train/Val Split in this Repository**:
  - We use a **stratified 90/10 split** on the training set (seed `42`):
    - **Train split**: 108,000 samples
    - **Validation split**: 12,000 samples
    - **Test split**: 7,600 samples (unseen official test set)

### Automatic Download & Caching
The dataset is downloaded automatically on first run via Hugging Face `datasets`:
```python
from datasets import load_dataset
ds = load_dataset("ag_news", cache_dir="data")
```
Fallback to `fancyzhx/ag_news` is implemented in `src/newsclf/data.py:load_raw_dataset()`.

---

## 2. Additional Evaluation Datasets Included

1. **`data/modern_headlines.csv`**:
   - 50 manually curated contemporary 2020s news headlines across World, Sports, Business, and Sci/Tech.
   - Evaluates **temporal drift** from 2004 AG News vocabulary to the modern era (e.g. *OpenAI, ChatGPT, CrowdStrike, Artemis, Starship*).

2. **`data/out_of_domain_headlines.csv`**:
   - 50 hand-crafted out-of-domain headlines spanning food, recipes, entertainment, celebrity gossip, pets, home gardening, and wellness.
   - Evaluates whether the **calibrated uncertainty threshold (τ = 0.60)** effectively triggers warning flags for non-news inputs.
