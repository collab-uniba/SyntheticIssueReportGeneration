import pandas as pd
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os

# ---------------------
# Argomenti da shell
# ---------------------
parser = argparse.ArgumentParser()
parser.add_argument('-p','--predictions', type=str, required=True, help='Path al file CSV di predizioni')
parser.add_argument('-t','--testset', type=str, required=True, help='Path al file CSV di test')
parser.add_argument('--output_dir', type=str, default='results', help='Percorso della cartella di output')

args = parser.parse_args()

# Creazione cartella output se non esiste
output_dir = args.output_dir
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Caricamento dataset
pred_df = pd.read_csv(args.predictions, delimiter=';', quotechar='"')
test_df = pd.read_csv(args.testset, delimiter=';', quotechar='"')

# Normalizza i nomi delle colonne
pred_df = pred_df.rename(columns={"ID": "id", "Prediction": "prediction"})
test_df = test_df.rename(columns={"ID": "id", "Polarity": "polarity"})

# Merge sui dati
merged_df = pd.merge(
    pred_df,
    test_df,
    on="id",
    how="inner",        # tipo di merge: inner, left, right, outer
    validate="1:1"      # assicura che ogni id sia unico in entrambe le tabelle
)

y_true = merged_df["polarity"]
y_pred = merged_df["prediction"]

# ---------------------
# Metriche globali
# ---------------------
accuracy = accuracy_score(y_true, y_pred)
precision_macro = precision_score(y_true, y_pred, average="macro")
recall_macro = recall_score(y_true, y_pred, average="macro")
f1_macro = f1_score(y_true, y_pred, average="macro")
f1_weighted = f1_score(y_true, y_pred, average="weighted")

report = classification_report(y_true, y_pred, output_dict=True)

results = {
    "accuracy": accuracy,
    "precision_macro": precision_macro,
    "recall_macro": recall_macro,
    "f1_macro": f1_macro,
    "f1_weighted": f1_weighted,

    "precision_positive": report["positive"]["precision"],
    "recall_positive": report["positive"]["recall"],
    "f1_positive": report["positive"]["f1-score"],

    "precision_negative": report["negative"]["precision"],
    "recall_negative": report["negative"]["recall"],
    "f1_negative": report["negative"]["f1-score"],

    "precision_neutral": report["neutral"]["precision"],
    "recall_neutral": report["neutral"]["recall"],
    "f1_neutral": report["neutral"]["f1-score"]
}

# ---------------------
# Salvataggio CSV
# ---------------------
output_file = os.path.join(args.output_dir, "metrics.csv")
pd.DataFrame([results]).to_csv(output_file, index=False)
print(f"Metriche salvate in: {output_file}")