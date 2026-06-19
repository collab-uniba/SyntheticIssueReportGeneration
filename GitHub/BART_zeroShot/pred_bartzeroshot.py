import argparse
import os
import json
import csv
import pandas as pd
import torch
import time
from sklearn.metrics import classification_report, accuracy_score
from transformers import pipeline
from pathlib import Path

# Verifica se CUDA è disponibile e stampa informazioni sulla GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device in uso:", device)

parser = argparse.ArgumentParser(description="BART Zero-Shot Classification")

parser.add_argument("-t", "--test_file", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="BART_zeroShot_outputs")

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "datasets"

# Carica il dataset di test
df = pd.read_csv(DATA_DIR / args.test_file, sep=";")

df = df.dropna(subset=["Text", "Polarity"])

test_ids = df["ID"].astype(str).tolist()
texts = df["Text"].astype(str).tolist()
true_labels = df["Polarity"].tolist()

device = 0 if torch.cuda.is_available() else -1
print("Pipeline device:", device)

# Zero-shot model
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device
)

candidate_labels = ["positive", "negative", "neutral"]

predictions = []
scores = []

start_time = time.time()

print("Running zero-shot inference...")

for i, text in enumerate(texts):
    result = classifier(text, candidate_labels)

    pred = result["labels"][0]
    score = result["scores"][0]

    predictions.append(pred)
    scores.append(score)

    if i % 100 == 0:
        print(f"Processed {i}/{len(texts)}")
        
inference_time = time.time() - start_time

memory_mb = (
    torch.cuda.max_memory_allocated() / 1024**2
    if torch.cuda.is_available()
    else 0
)

# Metriche
report = classification_report(true_labels, predictions, output_dict=True)

accuracy = accuracy_score(true_labels, predictions)

MACRO_AVG = "macro avg"

results = {
    "dataset": {
        "test": os.path.basename(args.test_file)
    },
    "info": {
        "model": "facebook/bart-large-mnli",
        "task": "zero-shot classification",
        "inference_time": inference_time,
        "gpu_memory_mb": memory_mb,
    },
    "overall_metrics": {
        "accuracy": accuracy,
        "precision_macro": report[MACRO_AVG]["precision"],
        "recall_macro": report[MACRO_AVG]["recall"],
        "f1_macro": report[MACRO_AVG]["f1-score"],
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

print(json.dumps(results, indent=4))

# Salva risultati metriche
with open(os.path.join(args.output_dir, "BART_zeroShot_metrics.json"), "w") as f:
    json.dump(results, f, indent=4)

# Salva predizioni
df_out = pd.DataFrame({
    "ID": test_ids,
    "TrueLabel": true_labels,
    "Prediction": predictions,
    "Text": texts,
})

df_out.to_csv(
    os.path.join(args.output_dir, "BART_zeroShot_predictions.csv"),
    index=False,
    sep=";",
    quotechar='"',
    quoting=csv.QUOTE_MINIMAL
)

print(f"Predizioni salvate in: {os.path.join(args.output_dir, 'BART_zeroShot_predictions.csv')}")

print(f"Metriche salvate in: {os.path.join(args.output_dir, 'BART_zeroShot_metrics.json')}")
