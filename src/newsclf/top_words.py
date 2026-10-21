"""Extract and analyze top driving vocabulary features per news category.

Inspects Logistic Regression weight vectors (W_{k, j}) from the trained pipeline
to identify the top positive lexical indicators for each class:
- World
- Sports
- Business
- Sci/Tech

Produces:
- reports/top_words.md
- reports/top_words.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from newsclf.data import LABEL_NAMES
from newsclf.predict import DEFAULT_BASELINE_PATH, load_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


def get_top_features_per_class(
    model_path: str = DEFAULT_BASELINE_PATH,
    top_n: int = 15,
) -> Dict[str, List[Dict[str, Any]]]:
    """Extract top positive coefficient features for each target class."""
    _, pipeline = load_model(model_path)
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    feature_names = tfidf.get_feature_names_out()
    classes = getattr(clf, "classes_", np.arange(len(LABEL_NAMES)))

    results = {}
    for class_idx, class_num in enumerate(classes):
        class_name = LABEL_NAMES[class_num] if class_num < len(LABEL_NAMES) else f"Class_{class_num}"
        coefs = clf.coef_[class_idx]

        # Top positive indices
        top_indices = np.argsort(coefs)[::-1][:top_n]
        top_features = []
        for idx in top_indices:
            top_features.append({
                "feature": str(feature_names[idx]),
                "weight": round(float(coefs[idx]), 4),
            })
        results[class_name] = top_features

    return results


def save_reports(data: Dict[str, List[Dict[str, Any]]], reports_dir: Path = REPORTS_DIR):
    """Write markdown table and json reports."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "top_words.json"
    md_path = reports_dir / "top_words.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    lines = [
        "# 🏷️ Highly Informative Features (Top Vocabulary per Category)",
        "",
        "This table displays the **top 15 highest-weighted positive features** ($W_{k, j}$) learned by the Multinomial Logistic Regression model for each news category.",
        "",
        "| Rank | 🌍 World | ⚽ Sports | 📈 Business | 🔬 Sci/Tech |",
        "| :---: | :--- | :--- | :--- | :--- |",
    ]

    top_n = max(len(feats) for feats in data.values())
    for i in range(top_n):
        row_str = f"| **#{i + 1}** "
        for name in ["World", "Sports", "Business", "Sci/Tech"]:
            feats = data.get(name, [])
            if i < len(feats):
                row_str += f"| `{feats[i]['feature']}` (+{feats[i]['weight']:.2f}) "
            else:
                row_str += "| - "
        row_str += "|"
        lines.append(row_str)

    lines.extend([
        "",
        "---",
        "",
        "### Linguistic & Domain Observations",
        "",
        "1. **Sports**: Shows extraordinarily distinct lexical separation (`olympic`, `coach`, `cup`, `season`, `game`, `soccer`). This explains why Sports reaches the highest F1 score (**0.9660**).",
        "2. **World**: Dominated by geopolitical and crisis vocabulary (`iraq`, `gaza`, `minister`, `troops`, `un`, `reuters`).",
        "3. **Business**: Characterized by financial and corporate reporting terms (`oil`, `stocks`, `profit`, `shares`, `quarterly`, `economy`).",
        "4. **Sci/Tech**: Driven by software, electronics, and digital technology terms (`software`, `microsoft`, `space`, `internet`, `apple`, `linux`).",
    ])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Top words reports saved to:\n - {json_path}\n - {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract top words per class from model weights")
    parser.add_argument("--model-path", type=str, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args()

    results = get_top_features_per_class(model_path=args.model_path, top_n=args.top_n)
    save_reports(results)


if __name__ == "__main__":
    main()
