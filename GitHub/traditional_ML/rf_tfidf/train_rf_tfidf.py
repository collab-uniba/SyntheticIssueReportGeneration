import argparse
import json
import pandas as pd
import os
import torch
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Verifica se CUDA è disponibile e stampa informazioni sulla GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device in uso:", device)

# Argomenti
parser = argparse.ArgumentParser(description="Random Forest + TF-IDF")

parser.add_argument("-d", "--train_file", required=True, help="Percorso al CSV di training.")
parser.add_argument("-t", "--test_file", required=True, help="Percorso al CSV di test.")

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
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1,2),
            max_features=10000,
            sublinear_tf=True
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
model.fit(X_train, y_train)

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
    "model": "Random Forest",
    "features": "TF-IDF",
    "n_estimators": 100,
    "ngrams": "1-2",

    "accuracy": accuracy_score(y_test, predictions),

    "precision": report[MACRO_AVG]["precision"],
    "recall": report[MACRO_AVG]["recall"],
    "f1_macro": report[MACRO_AVG]["f1-score"],
    "f1_weighted": report[WEIGHTED_AVG]["f1-score"]
}

# Output
OUTPUT_DIR = "rf_tfidf_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(json.dumps(results, indent=4))

predictions_df = pd.DataFrame({
    "ID": test_df["ID"],
    "TrueLabel": y_test,
    "Prediction": predictions,
    "Text": test_df["Text"]
})

predictions_df.to_csv(
    os.path.join(OUTPUT_DIR, "rf_tfidf_predictions.csv"),
    index=False,
    sep=";"
)

with open(os.path.join(OUTPUT_DIR, "rf_tfidf_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)