import argparse
import json
import pandas as pd
import os
import torch
import time
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report)
from pathlib import Path

# Verifica se CUDA è disponibile e stampa informazioni sulla GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device in uso:", device)

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--train_file", required=True, help="Percorso al file CSV di training.")
parser.add_argument("-t", "--test_file", required=True, help="Percorso al file CSV di test.")
parser.add_argument("-g", "--n_grams", type=int, default=2, help="Numero massimo di n-grammi da considerare.")
parser.add_argument("--output_dir", type=str, default="svm_bow_outputs", help="Directory dove salvare i risultati")
args = parser.parse_args()

# Caricamento dataset
train_df = pd.read_csv(args.train_file, sep=";")
test_df = pd.read_csv(args.test_file, sep=";")

# Rimuovi valori mancanti
train_df = train_df.dropna(subset=["Text", "Polarity"])
test_df = test_df.dropna(subset=["Text", "Polarity"])
 
# Labels: 0 = negative, 1 = neutral, 2 = positive
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
        "svm",
        SVC(
            kernel="linear",
            C=1.0,
            gamma="scale",
            random_state=42
        )
    )
])

# Train
start_time = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start_time

memory_mb = (
    torch.cuda.max_memory_allocated() / 1024**2
    if torch.cuda.is_available()
    else 0
)

# Predizioni
predictions = model.predict(X_test)

# Metriche di valutazione
accuracy = accuracy_score(y_test, predictions)

report = classification_report(
    y_test,
    predictions,
    output_dict=True
)

# Risultati
MACRO_AVG = "macro avg"
WEIGHTED_AVG = "weighted avg"

results = {
    "dataset": {
        "train": os.path.basename(args.train_file),
        "test": os.path.basename(args.test_file)
    },
    "info": {
        "model": "SVM",
        "kernel": "linear",
        "features": "Bag of Words",
        "ngrams": f"1-{args.n_grams}",
        "training_time_seconds": training_time,
        "peak_gpu_memory_mb": memory_mb
    },
    "overall_metrics": {
        "accuracy": accuracy,
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

# Crea cartella output
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Stampa risultati
print(json.dumps(results, indent=4))

# Salvataggio predictions su file CSV
predictions_df = pd.DataFrame({
    "ID": test_df["ID"],
    "TrueLabel": y_test,
    "Prediction": predictions,
    "Text": test_df["Text"]
})

predictions_df.to_csv(
    os.path.join(OUTPUT_DIR, "svm_bow_predictions.csv"),
    index=False,
    sep=";"
)

# Salvataggio risultati su file JSON
with open(os.path.join(OUTPUT_DIR, "svm_bow_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)