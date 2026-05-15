import argparse
import pandas as pd
import os
import csv
import json
import torch
from datasets import load_dataset, DatasetDict, Features, Value
from typing import cast
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Verifica se CUDA è disponibile e stampa informazioni sulla GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device in uso:", device)

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--train_file", type=str, required=True, help="Percorso al file CSV di training.")
parser.add_argument("-t", "--test_file", type=str, default=None, help="Percorso al file CSV di test (opzionale).")
parser.add_argument("-n", "--num_samples", type=int, default=20, help="Numero di sample per etichetta da usare per il training. 0 = tutto il dataset.")
parser.add_argument("-s", "--split_ratio", type=float, default=0.3, help="Percentuale del dataset di training da usare come test se non è fornito un test set.")
parser.add_argument("--output_dir", type=str, default="outputs", help="Directory dove salvare i risultati")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# Definisci le feature per forzare tutti i tipi
features = Features({
    "ID": Value("string"),
    "Polarity": Value("string"),
    "Text": Value("string")
})

data_files = {"train": args.train_file}
if args.test_file:
    data_files["test"] = args.test_file

dataset = load_dataset(
    "csv",
    data_files=data_files,
    delimiter=";",
    features=features
)

# Se non c'è test set, fai uno split del train
if "test" not in dataset:
    df = dataset["train"].to_pandas()
    
    df_train, df_test = train_test_split(df, test_size=args.split_ratio, stratify=df["Polarity"], random_state=42)
    
    dataset = DatasetDict({
        "train": dataset["train"].from_pandas(df_train.reset_index(drop=True)),
        "test": dataset["train"].from_pandas(df_test.reset_index(drop=True))
    })

# Mescola il training set (importante per few-shot)
dataset["train"] = dataset["train"].shuffle(seed=42)

# per usare tutto il dataset e non fare un sample passare come parametro n<=0
if args.num_samples > 0:
    train_dataset = sample_dataset(dataset["train"], label_column="Polarity", num_samples=args.num_samples)
else:
    train_dataset = dataset["train"]

test_dataset = dataset["test"]

# Forza Text a stringa (evita errori di tokenizzazione)
train_dataset = train_dataset.map(lambda x: {"Text": str(x["Text"])})
test_dataset = test_dataset.map(lambda x: {"Text": str(x["Text"])})

model = SetFitModel.from_pretrained(
    "all-mpnet-base-v2",
    labels=["negative", "positive", "neutral"],
)

training_args = TrainingArguments(
    batch_size=16,
    num_epochs=4,
    evaluation_strategy="no", # in conflitto con load_best_model_at_end=True, che richiede evaluation_strategy diverso da "no", ma altrimenti non salva il modello migliore
    save_strategy="no",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    metric="accuracy",
    column_mapping={"Text": "text", "Polarity": "label"}
)

print("Training starting...")

trainer.train()

# Predictions
test_ids = test_dataset["ID"]
test_texts = test_dataset["Text"]
true_labels = test_dataset["Polarity"]
predicted_labels = model.predict(test_texts).tolist()

# Metriche
report = classification_report(
    true_labels,
    predicted_labels,
    output_dict=True
)

MACRO_AVG = "macro avg"
WEIGHTED_AVG = "weighted avg"

results = {
    "datasets": {
        "train": args.train_file,  # Path(args.train_file).name,
        "test": args.test_file if args.test_file else "split from train with ratio " + str(args.split_ratio)  # Path(args.test_file).name if args.test_file else "split from train with ratio " + str(args.split_ratio)
    },
    "model": "SetFit",
    "base_model": "all-mpnet-base-v2",
    "num_samples": args.num_samples,
    
    "accuracy": accuracy_score(true_labels, predicted_labels),
    "precision": report[MACRO_AVG]["precision"],
    "recall": report[MACRO_AVG]["recall"],
    "f1_macro": report[MACRO_AVG]["f1-score"],
    "f1_weighted": report[WEIGHTED_AVG]["f1-score"],
}

print(json.dumps(results, indent=4))

# Salvataggio risultati
df_results = pd.DataFrame({
    "ID": test_ids,
    "TrueLabel": true_labels,
    "Prediction": predicted_labels,
    "Text": test_texts,
})

df_results.to_csv(
    os.path.join(args.output_dir, "SetFit_predictions.csv"),
    index=False,
    sep=';'           # separatore ;
)

print(f"Predizioni salvate in: {os.path.join(args.output_dir, 'SetFit_predictions.csv')}")

with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
    json.dump(results, f, indent=4)

print(f"Metriche salvate in: {os.path.join(args.output_dir, 'metrics.json')}")
