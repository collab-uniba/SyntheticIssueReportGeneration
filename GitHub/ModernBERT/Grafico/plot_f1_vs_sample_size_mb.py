import sys
import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from kneed import KneeLocator

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# -------------------
# DEVICE
# -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# -------------------
# INPUT
# -------------------
train_csv_path = sys.argv[1]
test_csv_path = sys.argv[2]

dataset = load_dataset(
    "csv",
    data_files={"train": train_csv_path, "test": test_csv_path},
    delimiter=";",
    quotechar='"'
)

# -------------------
# SETTINGS
# -------------------
sample_sizes = [5, 10, 15, 20, 25, 50, 100, 200, "all_capped"]

model_name = "answerdotai/ModernBERT-base"

labels = ["negative", "positive", "neutral"]
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(model_name)

# -------------------
# LABEL ENCODING
# -------------------
def encode_labels(example):
    example["labels"] = label2id[example["Polarity"]]
    return example

# -------------------
# TOKENIZATION
# -------------------
def preprocess(example):
    return tokenizer(
        example["Text"],
        truncation=True,
        padding=False,
        max_length=256
    )

# -------------------
# TEST SET (static)
# -------------------
test_dataset = dataset["test"]

# -------------------
# OUTPUT DIR
# -------------------
output_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ModernBERT_outputs"
)
os.makedirs(output_dir, exist_ok=True)

results = []

# -------------------
# MAIN LOOP
# -------------------
for size in sample_sizes:
    print(f"\n=== Training size: {size}")

    train_base = dataset["train"].shuffle(seed=42)

    capped_size = min(len(train_base), 1000)
    train_base = train_base.select(range(capped_size))

    if size == "all_capped":
        train_dataset = train_base
    else:
        train_dataset = train_base.shuffle(seed=42).select(range(size))

    # -------------------
    # LABEL ENCODING FIRST
    # -------------------
    train_dataset = train_dataset.map(encode_labels)
    test_tok = test_dataset.map(encode_labels)

    # -------------------
    # TOKENIZATION
    # -------------------
    train_dataset = train_dataset.map(preprocess)
    test_tok = test_tok.map(preprocess)

    # -------------------
    # REMOVE OLD COLUMNS
    # -------------------
    train_dataset = train_dataset.remove_columns(
        [c for c in train_dataset.column_names if c not in ["input_ids", "attention_mask", "labels"]]
    )

    test_tok = test_tok.remove_columns(
        [c for c in test_tok.column_names if c not in ["input_ids", "attention_mask", "labels"]]
    )

    # -------------------
    # FORMAT
    # -------------------
    train_dataset.set_format("torch")
    test_tok.set_format("torch")

    # -------------------
    # MODEL
    # -------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    ).to(device)

    # -------------------
    # TRAINING ARGS
    # -------------------
    args = TrainingArguments(
        output_dir="./tmp",
        per_device_train_batch_size=16,
        num_train_epochs=5,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=10,
        fp16=torch.cuda.is_available()
    )

    # -------------------
    # METRICS
    # -------------------
    def compute_metrics(eval_pred):
        logits = eval_pred.predictions
        labels_ = eval_pred.label_ids
        preds = logits.argmax(axis=1)

        return {
            "f1": f1_score(labels_, preds, average="macro"),
            "accuracy": accuracy_score(labels_, preds),
            "precision": precision_score(labels_, preds, average="macro", zero_division=0),
            "recall": recall_score(labels_, preds, average="macro", zero_division=0),
        }

    # -------------------
    # TRAINER
    # -------------------
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_tok,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )

    trainer.train()

    # -------------------
    # PREDICTIONS
    # -------------------
    preds = trainer.predict(test_tok)
    y_pred = preds.predictions.argmax(axis=1)
    y_true = preds.label_ids

    f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    results.append({
        "sample_size": capped_size if size == "all_capped" else size,
        "f1_score": f1,
        "accuracy": acc,
        "precision": prec,
        "recall": rec
    })

    print(f"F1: {f1:.4f} | Acc: {acc:.4f}")

# -------------------
# ANALYSIS
# -------------------
df = pd.DataFrame(results)
df["sample_size"] = df["sample_size"].astype(int)
df = df.sort_values("sample_size")

x = df["sample_size"].values
y = df["f1_score"].values

kneedle = KneeLocator(x, y, curve="concave", direction="increasing")
knee_x = kneedle.knee
knee_y = kneedle.knee_y

plt.figure(figsize=(10, 6))
plt.xscale("log")
plt.plot(x, y, marker="o")

if knee_x is not None:
    plt.axvline(knee_x, linestyle="--", color="red")
    plt.scatter(knee_x, knee_y, color="red")
    plt.text(knee_x, knee_y, f"Knee: {knee_x}")

plt.title("ModernBERT - F1 vs Sample Size")
plt.xlabel("Training samples")
plt.ylabel("Macro F1")
plt.grid()
plot_path = os.path.join(output_dir, "f1_score_vs_sample_size.png")

plt.tight_layout()
plt.savefig(plot_path, dpi=300, bbox_inches="tight")

plt.show()

print("Plot salvato in:", plot_path)
print("Knee:", knee_x, knee_y)