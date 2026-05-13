import json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report)

# Caricamento dataset
train_df = pd.read_csv("train_github.csv", sep=";")
test_df = pd.read_csv("test_github.csv", sep=";")

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
model.fit(X_train, y_train)

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
    "model": "SVM",
    "kernel": "linear",
    "features": "TF-IDF",
    "ngrams": "1-2",
    "accuracy": accuracy,

    "precision": report[MACRO_AVG]["precision"],
    "recall": report[MACRO_AVG]["recall"],
    "f1_macro": report[MACRO_AVG]["f1-score"],
    "f1_weighted": report[WEIGHTED_AVG]["f1-score"]
}

# Stampa risultati
print(json.dumps(results, indent=4))

# Salvataggio risultati su file JSON
with open("svm_tfidf_results.json", "w") as f:
    json.dump(results, f, indent=4)