import argparse
import os
import csv
import json
import time
import torch
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict, Features, Value
from typing import cast
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# Verifica se CUDA è disponibile e stampa informazioni sulla GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device in uso: {device}")

parser = argparse.ArgumentParser(description="ModernBERT fine-tuning")

parser.add_argument("-d", "--train_file", type=str, required=True, help="Percorso al file CSV di training.")
parser.add_argument("-t", "--test_file", type=str, default=None, help="Percorso al file CSV di test.")
parser.add_argument("-s", "--split_ratio", type=float, default=0.3, help="Percentuale del dataset usata come test se test_file non è fornito.")
parser.add_argument("-n", "--num_samples", type=int, default=100, help="Numero di sample per etichetta da usare per il training. 0 = tutto il dataset.")
parser.add_argument("--output_dir", type=str, default="outputs", help="Directory di output.")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
parser.add_argument("--max_length", type=int, default=256, help="Lunghezza massima token.")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
os.makedirs(checkpoints_dir, exist_ok=True)

# Definisci le feature per forzare tutti i tipi
features = Features({
    "ID": Value("string"),
    "Polarity": Value("string"),
    "Text": Value("string")
})

# Labels
label2id = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

id2label = {v: k for k, v in label2id.items()} #dinamico in caso di più classi o classi diverse, non hardcodato

# Caricamento dataset
data_files = {"train": args.train_file}

if args.test_file:
    data_files["test"] = args.test_file

dataset = load_dataset(
    "csv",
    data_files=data_files,
    delimiter=";",
    features=features
)

# Train-test split se non c'è test set
if "test" not in dataset:
    df = dataset["train"].to_pandas()

    df_train, df_test = train_test_split(df, test_size=args.split_ratio, stratify=df["Polarity"], random_state=42)

    dataset = DatasetDict({
        "train": Dataset.from_pandas(df_train.reset_index(drop=True)),
        "test": Dataset.from_pandas(df_test.reset_index(drop=True))
    })

# Tokenizer
def encode_labels(example):
    example["label"] = label2id[example["Polarity"]]
    return example

dataset = dataset.map(encode_labels)

model_name = "answerdotai/ModernBERT-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

def tokenize_function(batch):
    return tokenizer(
        batch["Text"],
        truncation=True,
        padding=False,
        max_length=args.max_length
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

# Metriche
def compute_metrics(eval_pred):

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro"
    )

    accuracy = accuracy_score(labels, predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Training Arguments
training_args = TrainingArguments(
    output_dir=checkpoints_dir,
    learning_rate=2e-5,
    seed=42,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Training
print("Training starting...")
start_time = time.time()

trainer.train()

training_time = time.time() - start_time

print(f"Training time: {training_time:.2f} seconds")

# Memoria GPU usata
memory_mb = (
    torch.cuda.max_memory_allocated() / 1024**2
    if torch.cuda.is_available()
    else 0
)

print(f"Peak GPU memory usage: {memory_mb:.2f} MB")

# Predictions
predictions = trainer.predict(tokenized_dataset["test"])
predicted_classes = np.argmax(predictions.predictions, axis=-1)

true_classes = predictions.label_ids

predicted_labels = [
    id2label[p]
    for p in predicted_classes
]

true_labels = [
    id2label[t]
    for t in true_classes
]

# Classification Report
report = classification_report(
    true_labels,
    predicted_labels,
    digits=4,
    output_dict=True
)

print("\nClassification Report:")
print(classification_report(
    true_labels,
    predicted_labels,
    digits=4
))

# Salva risultati in JSON
MACRO_AVG = "macro avg"
WEIGHTED_AVG = "weighted avg"

metrics_output = {
    "datasets": {
        "train": os.path.basename(args.train_file),
        "test": os.path.basename(args.test_file) if args.test_file else f"split_from_train_ratio_{args.split_ratio}"
    },
    "info": {
        "model": "ModernBERT",
        "base_model": model_name,
        "training_time_seconds": training_time,
        "gpu_memory_mb": memory_mb
    },
    "overall_metrics": {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": report[MACRO_AVG]["precision"],
        "recall": report[MACRO_AVG]["recall"],
        "f1_macro": report[MACRO_AVG]["f1-score"],
        "f1_weighted": report[WEIGHTED_AVG]["f1-score"]
    },
    "class_positive": {
        "precision": report["positive"]["precision"],
        "recall": report["positive"]["recall"],
        "f1": report["positive"]["f1-score"],
        "samples": report["positive"]["support"]
    },
    "class_neutral": {
        "precision": report["neutral"]["precision"],
        "recall": report["neutral"]["recall"],
        "f1": report["neutral"]["f1-score"],
        "samples": report["neutral"]["support"]
    },
    "class_negative": {
        "precision": report["negative"]["precision"],
        "recall": report["negative"]["recall"],
        "f1": report["negative"]["f1-score"],
        "samples": report["negative"]["support"]
    }
}

metrics_path = os.path.join(
    args.output_dir,
    "metrics.json"
)

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics_output, f, indent=2)

print(f"Metriche salvate in: {metrics_path}")

# Salva predizioni in csv
test_dataset = dataset["test"]

if "ID" in test_dataset.column_names:
    test_ids = test_dataset["ID"]
else:
    test_ids = list(range(len(test_dataset)))

test_texts = test_dataset["Text"]

df_results = pd.DataFrame({
    "ID": test_ids,
    "TrueLabel": true_labels,
    "Prediction": predicted_labels,
    "Text": test_texts
})

output_path = os.path.join(
    args.output_dir,
    "ModernBERT_predictions.csv"
)

df_results.to_csv(
    output_path,
    index=False,
    sep=';',
    quotechar='"',
    quoting=csv.QUOTE_MINIMAL
)

print(f"Predizioni salvate in: {output_path}")