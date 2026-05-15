import json
import pandas as pd
import os
import torch
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

# Verifica se CUDA è disponibile e stampa informazioni sulla GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device in uso:", device)

# Caricamento dataset
BASE_DIR = Path(__file__).resolve().parents[2]  # Directory del progetto
DATA_DIR = BASE_DIR / "datasets"

train_path = DATA_DIR / "train_github.csv"
test_path = DATA_DIR / "test_github.csv"

train_df = pd.read_csv(train_path, sep=";")
test_df = pd.read_csv(test_path, sep=";")

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
model.fit(X_train, y_train)

# Predizioni
predictions = model.predict(X_test)

# Metriche di valutazione
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
        "train": train_path.name,
        "test": test_path.name
    },
    "model": "SVM",
    "kernel": "linear",
    "features": "TF-IDF",
    "ngrams": "1-2",
    
    "accuracy": accuracy_score(y_test, predictions),
    "precision": report[MACRO_AVG]["precision"],
    "recall": report[MACRO_AVG]["recall"],
    "f1_macro": report[MACRO_AVG]["f1-score"],
    "f1_weighted": report[WEIGHTED_AVG]["f1-score"]
}

# Crea cartella output
OUTPUT_DIR = "svm_tfidf_outputs"
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
    os.path.join(OUTPUT_DIR, "svm_tfidf_predictions.csv"),
    index=False,
    sep=";"
)

# Salvataggio risultati su file JSON
with open(os.path.join(OUTPUT_DIR, "svm_tfidf_results.json"), "w") as f:
    json.dump(results, f, indent=4)