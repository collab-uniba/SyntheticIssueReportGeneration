import argparse
import os
import json
import csv
import pandas as pd
import torch
from sklearn.metrics import classification_report, accuracy_score
from transformers import pipeline

# Verifica se CUDA è disponibile e stampa informazioni sulla GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device in uso:", device)

parser = argparse.ArgumentParser(description="BART Zero-Shot Classification")

parser.add_argument("-t", "--test_file", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="BART_zeroShot_outputs")

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# Carica il dataset di test
df = pd.read_csv(args.test_file, sep=";")

df = df.dropna(subset=["Text", "Polarity"])

texts = df["Text"].astype(str).tolist()
true_labels = df["Polarity"].tolist()

# Zero-shot model
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device
)

candidate_labels = ["positive", "negative", "neutral"]

predictions = []
scores = []

print("Running zero-shot inference...")

for i, text in enumerate(texts):
    result = classifier(text, candidate_labels)

    pred = result["labels"][0]
    score = result["scores"][0]

    predictions.append(pred)
    scores.append(score)

    if i % 100 == 0:
        print(f"Processed {i}/{len(texts)}")

# Metriche
report = classification_report(true_labels, predictions, output_dict=True)

accuracy = accuracy_score(true_labels, predictions)

MACRO_AVG = "macro avg"

results = {
    "model": "facebook/bart-large-mnli",
    "task": "zero-shot classification",
    "accuracy": accuracy,
    "precision_macro": report[MACRO_AVG]["precision"],
    "recall_macro": report[MACRO_AVG]["recall"],
    "f1_macro": report[MACRO_AVG]["f1-score"]
}

print(json.dumps(results, indent=4))

# Salva risultati metriche
with open(os.path.join(args.output_dir, "BART_zeroShot_metrics.json"), "w") as f:
    json.dump(results, f, indent=4)

# Salva predizioni
df_out = pd.DataFrame({
    "Text": texts,
    "TrueLabel": true_labels,
    "Prediction": predictions,
    "Score": scores
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
