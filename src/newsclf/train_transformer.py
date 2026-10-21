"""Fine-tune distilbert-base-uncased on AG News using Hugging Face Trainer."""

import argparse
import os
import sys
from typing import Dict

import numpy as np
from datasets import Dataset

from newsclf.data import ID2LABEL, LABEL2ID, LABEL_NAMES, prepare_datasets

DEFAULT_OUTPUT_DIR = os.path.join("models", "transformer_model")
MODEL_CHECKPOINT = "distilbert-base-uncased"


def compute_metrics(eval_pred):
    """Compute accuracy and macro F1 for Hugging Face Trainer."""
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
    }


def train_transformer(
    model_name_or_path: str = MODEL_CHECKPOINT,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    epochs: int = 3,
    batch_size: int = 16,
    max_length: int = 128,
    learning_rate: float = 2e-5,
    subset_size: int = 20000,
    seed: int = 42,
):
    """Fine-tune DistilBERT sequence classifier using Hugging Face Trainer.

    Args:
        model_name_or_path: Pretrained Hugging Face checkpoint name.
        output_dir: Path to directory where the fine-tuned model will be saved.
        epochs: Number of training epochs (default: 3).
        batch_size: Training and evaluation batch size.
        max_length: Maximum sequence length for tokenization (default: 128).
        learning_rate: Initial AdamW learning rate (default: 2e-5).
        subset_size: Optional subset size for training on CPU/laptops (0 for full 120k).
        seed: Random seed for reproducibility.
    """
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as e:
        print(f"Error importing PyTorch or Transformers: {e}")
        print("Please install torch and transformers: pip install torch transformers")
        sys.exit(1)

    print(f"Loading data (lowercase=False for transformer, seed={seed})...")
    train_df, val_df, test_df = prepare_datasets(lowercase=False, random_state=seed)

    if subset_size > 0 and len(train_df) > subset_size:
        print(f"Subsetting training set to {subset_size} samples for fast laptop training...")
        train_df = train_df.sample(n=subset_size, random_state=seed).reset_index(drop=True)
        val_subset = int(subset_size * 0.1)
        val_df = val_df.sample(n=min(val_subset, len(val_df)), random_state=seed).reset_index(drop=True)

    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}, Test samples: {len(test_df)}")

    print(f"Initializing tokenizer for '{model_name_or_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    train_ds = Dataset.from_pandas(train_df).map(tokenize_batch, batched=True)
    val_ds = Dataset.from_pandas(val_df).map(tokenize_batch, batched=True)
    test_ds = Dataset.from_pandas(test_df).map(tokenize_batch, batched=True)

    print(f"Loading model '{model_name_or_path}' with 4 classes...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=4,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        seed=seed,
        logging_steps=100,
        report_to="none",
        use_cpu=not torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Beginning fine-tuning...")
    trainer.train()

    print("Evaluating on test split...")
    test_metrics = trainer.evaluate(test_ds)
    print(f"Test Set Metrics: {test_metrics}")

    print(f"Saving fine-tuned model and tokenizer to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Transformer training completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT sequence classifier on AG News")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--max-len", type=int, default=128, help="Max sequence length (default: 128)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--subset-size", type=int, default=20000, help="Subset size for training (0 for full)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_transformer(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_len,
        learning_rate=args.lr,
        subset_size=args.subset_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
