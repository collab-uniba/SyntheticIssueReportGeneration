import argparse
import os
import time
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from datasets import load_dataset, Dataset, DatasetDict, Features, Value
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# -----------------------
# ARGUMENTS
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--train_file", type=str, required=True)
parser.add_argument("-t", "--test_file", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="scaling_outputs")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# -----------------------
# LOAD DATA
# -----------------------
features = Features({
    "ID": Value("string"),
    "Polarity": Value("string"),
    "Text": Value("string")
})

dataset = load_dataset(
    "csv",
    data_files={"train": args.train_file, "test": args.test_file},
    delimiter=";",
    features=features
)

label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {v: k for k, v in label2id.items()}

def encode(example):
    example["label"] = label2id[example["Polarity"]]
    return example

dataset = dataset.map(encode)

# -----------------------
# TOKENIZER / MODEL
# -----------------------
model_name = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["Text"], truncation=True, padding=False, max_length=256)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# -----------------------
# SUBSAMPLING FUNCTION
# -----------------------
def sample_per_class(df, n):
    return (
        df.groupby("Polarity", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), n), random_state=42))
        .reset_index(drop=True)
    )

# -----------------------
# TRAIN FUNCTION
# -----------------------
def train_and_eval(train_df, test_dataset):
    train_dataset = Dataset.from_pandas(train_df)

    tokenized_train = train_dataset.map(tokenize, batched=True)
    tokenized_test = test_dataset.map(tokenize, batched=True)

    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="tmp",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="no",
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"f1": f1_score(labels, preds, average="macro")}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    preds = trainer.predict(tokenized_test)
    y_pred = np.argmax(preds.predictions, axis=-1)
    y_true = preds.label_ids

    return f1_score(y_true, y_pred, average="macro")

# -----------------------
# EXPERIMENT SETTINGS
# -----------------------
samples_list = [20, 50, 100, 200, 400, 800, 1000, 1500, 2000, 4000]
results = []

test_df = dataset["test"].to_pandas()

# -----------------------
# RUN EXPERIMENT
# -----------------------
for n in samples_list:
    print(f"\nTraining with {n} samples per class")

    train_df = dataset["train"].to_pandas()
    train_df = sample_per_class(train_df, n)

    f1 = train_and_eval(train_df, dataset["test"])

    results.append((n, f1))

    print(f"Samples: {n} | F1: {f1:.4f}")

# -----------------------
# KNEE POINT (simple heuristic)
# -----------------------
x = np.array([r[0] for r in results])
y = np.array([r[1] for r in results])

second_derivative = np.diff(y, 2)
knee_index = np.argmin(second_derivative) + 1
knee_point = x[knee_index]

print(f"\nKnee point estimated at: {knee_point}")

# -----------------------
# PLOT
# -----------------------
plt.figure()
plt.plot(x, y, marker="o")
plt.axvline(knee_point, linestyle="--", label=f"Knee: {knee_point}")
plt.xlabel("Samples per class")
plt.ylabel("F1 Macro")
plt.title("ModernBERT Scaling Curve")
plt.legend()

plot_path = os.path.join(args.output_dir, "scaling_curve.png")
plt.savefig(plot_path)

print(f"Plot saved to: {plot_path}")

# -----------------------
# SAVE RESULTS
# -----------------------
with open(os.path.join(args.output_dir, "results.json"), "w") as f:
    json.dump({
        "results": results,
        "knee_point": int(knee_point)
    }, f, indent=2)