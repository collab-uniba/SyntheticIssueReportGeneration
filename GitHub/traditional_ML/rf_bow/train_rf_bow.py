import argparse
import json
import pandas as pd
import os
import torch
import time
import resource
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Argomenti
parser = argparse.ArgumentParser(description="Random Forest + Bag of Words")

parser.add_argument("-d", "--train_file", required=True, help="Percorso al CSV di training.")
parser.add_argument("-t", "--test_file", required=True, help="Percorso al CSV di test.")
parser.add_argument("-g", "--n_grams", type=int, default=2, help="Numero massimo di n-grammi da considerare.")
parser.add_argument("--output_dir", type=str, default="rf_bow_outputs", help="Directory dove salvare i risultati")

args = parser.parse_args()

# Caricamento del dataset
train_df = pd.read_csv(args.train_file, sep=";")
test_df = pd.read_csv(args.test_file, sep=";")

# Rimuovi valori mancanti
train_df = train_df.dropna(subset=["Text", "Polarity"])
test_df = test_df.dropna(subset=["Text", "Polarity"])

X_train = train_df["Text"]
y_train = train_df["Polarity"]

X_test = test_df["Text"]
y_test = test_df["Polarity"]

# Pipeline
model = Pipeline([ # NOSONAR
    (
        "bow",
        CountVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, args.n_grams),
            max_features=10000
        )
    ),
    (
        "rf",
        RandomForestClassifier(
            n_estimators=100,
            max_features="sqrt",
             min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
    )
])

# Training
print("Training starting...")
start_time = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start_time

peak_ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

# Predictions
predictions = model.predict(X_test)

# Metrics
report = classification_report(
    y_test,
    predictions,
    output_dict=True
)

MACRO_AVG = "macro avg"
WEIGHTED_AVG = "weighted avg"

results = {
    "dataset": {
        "train": os.path.basename(args.train_file),
        "test": os.path.basename(args.test_file)
    },
    "info": {
        "model": "Random Forest",
        "features": "Bag of Words",
        "n_estimators": 100,
        "ngrams": f"1-{args.n_grams}",
        "training_time_seconds": training_time,
        "peak_ram_mb": peak_ram_mb
    },
    "overall_metrics": {
        "accuracy": accuracy_score(y_test, predictions),
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

# Output
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(json.dumps(results, indent=4))

predictions_df = pd.DataFrame({
    "ID": test_df["ID"],
    "TrueLabel": y_test,
    "Prediction": predictions,
    "Text": test_df["Text"]
})

predictions_df.to_csv(
    os.path.join(OUTPUT_DIR, "rf_bow_predictions.csv"),
    index=False,
    sep=";"
)

with open(os.path.join(OUTPUT_DIR, "rf_bow_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)