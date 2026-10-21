"""Command Line Interface for News Category Classifier."""

import argparse
import json
import sys
from typing import Optional

import pandas as pd

from newsclf.predict import predict, predict_batch


def main(args: Optional[list] = None):
    parser = argparse.ArgumentParser(
        prog="newsclf",
        description="News Category Classifier CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Predict subcommand
    predict_parser = subparsers.add_parser("predict", help="Predict category for a single text")
    predict_parser.add_argument("text", type=str, help="Headline or news text to classify")
    predict_parser.add_argument("--model-path", type=str, default=None, help="Custom model path")
    predict_parser.add_argument("--json", action="store_true", help="Output raw JSON")

    # Batch prediction subcommand
    batch_parser = subparsers.add_parser("batch", help="Batch predict on CSV file")
    batch_parser.add_argument("input_csv", type=str, help="Path to input CSV file")
    batch_parser.add_argument("--text-col", type=str, default="text", help="Name of text column")
    batch_parser.add_argument("--output-csv", type=str, default="predictions.csv", help="Path to save output CSV")
    batch_parser.add_argument("--model-path", type=str, default=None, help="Custom model path")

    parsed_args = parser.parse_args(args)

    if parsed_args.command == "predict":
        res = predict(parsed_args.text, model_path=parsed_args.model_path)
        if parsed_args.json:
            print(json.dumps(res, indent=2))
        else:
            print("\n" + "=" * 40)
            print(f" Input:      {parsed_args.text}")
            print(f" Category:   {res['label']}")
            print(f" Confidence: {res['confidence'] * 100:.2f}%")
            print("-" * 40)
            print(" Class Probabilities:")
            for cat, prob in res["top_k_probs"].items():
                bar = "#" * int(prob * 25)
                print(f"  {cat:<10} {prob * 100:5.1f}% | {bar}")
            print("=" * 40 + "\n")
    elif parsed_args.command == "batch":
        df = pd.read_csv(parsed_args.input_csv)
        if parsed_args.text_col not in df.columns:
            print(f"Error: Column '{parsed_args.text_col}' not found in {parsed_args.input_csv}")
            sys.exit(1)
        texts = df[parsed_args.text_col].astype(str).tolist()
        predictions = predict_batch(texts, model_path=parsed_args.model_path)
        df["predicted_label"] = [p["label"] for p in predictions]
        df["confidence"] = [p["confidence"] for p in predictions]
        df.to_csv(parsed_args.output_csv, index=False)
        print(f"Saved {len(df)} predictions to {parsed_args.output_csv}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
