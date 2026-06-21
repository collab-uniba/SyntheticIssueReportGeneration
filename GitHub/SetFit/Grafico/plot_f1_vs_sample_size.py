import sys
import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import load_dataset, Dataset
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from kneed import KneeLocator

# Verifica se CUDA è disponibile e stampa informazioni sulla GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device in uso:", device)

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "datasets"

if len(sys.argv) != 3:
    sys.exit(1)

train_csv_path = DATASET_DIR / sys.argv[1]
test_csv_path = DATASET_DIR / sys.argv[2]

dataset = load_dataset(
    "csv",
    data_files={
        "train": train_csv_path,
        "test": test_csv_path
    },
    delimiter=";",
    quotechar='"'
)

sample_sizes = [5, 10, 15, 20, 25, 50, 100, 200, "all_capped"]

test_dataset = dataset["test"]
test_texts = test_dataset["Text"]
test_labels = test_dataset["Polarity"]
test_ids = test_dataset["ID"]

results = []

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Grafico_outputs")
os.makedirs(output_dir, exist_ok=True)

model_base = "all-mpnet-base-v2"
labels = ["negative", "positive", "neutral"]

for size in sample_sizes:
    print(f"\n===> Training with {size} samples")

    train_base = dataset["train"].shuffle(seed=42)

    capped_size = min(len(train_base), 1000)
    train_base = train_base.select(range(capped_size))

    if size == "all_capped":
        train_dataset = train_base
        actual_size = capped_size
    else:
        train_dataset = sample_dataset(
            train_base,
            label_column="Polarity",
            num_samples=size,
            seed=42
        )
        actual_size = size

    model = SetFitModel.from_pretrained(
        model_base,
        labels=labels,
    )

    args = TrainingArguments(
        batch_size=16,
        num_epochs=2,
        evaluation_strategy="no",
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        metric="accuracy",
        column_mapping={"Text": "text", "Polarity": "label"}
    )

    trainer.train()

    predicted_labels = model.predict(test_texts)

    f1 = f1_score(test_labels, predicted_labels, average="macro")
    acc = accuracy_score(test_labels, predicted_labels)
    precision = precision_score(test_labels, predicted_labels, average="macro", zero_division=0)
    recall = recall_score(test_labels, predicted_labels, average="macro", zero_division=0)

    metrics = {
        "sample_size": actual_size,
        "f1_score": f1,
        "accuracy": acc,
        "precision": precision,
        "recall": recall
    }

    results.append(metrics)

    size_label = f"all_capped_{actual_size}" if size == "all_capped" else str(size)
    metrics_path = os.path.join(output_dir, f"metrics_sample_{size_label}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    df_predictions = pd.DataFrame({
        "ID": test_ids,
        "Text": test_texts,
        "TrueLabel": test_labels,
        "PredictedLabel": predicted_labels
    })
    pred_path = os.path.join(output_dir, f"predictions_sample_{size_label}.csv")
    df_predictions.to_csv(pred_path, index=False, encoding="utf-8")

    print(f"F1-score: {f1:.4f} | Accuracy: {acc:.4f}")

# crea dataframe
df_results = pd.DataFrame(results)

# ordina per sample size
df_results = df_results.sort_values("sample_size")

x = df_results["sample_size"].values
y = df_results["f1_score"].values

# trova il knee
kneedle = KneeLocator(x, y, curve="concave", direction="increasing", S=1.0)

knee_x = kneedle.knee
knee_y = kneedle.knee_y

# plot
xticks = [5, 10, 15, 20, 25, 50, 100, 200, 1000]

plt.figure(figsize=(10, 6))
plt.xscale("log")
plt.plot(df_results["sample_size"], df_results["f1_score"], marker='o', label="F1-score (macro)")
plt.title("F1-score vs Sample Size")
plt.xlabel("Numero di campioni nel training set")
plt.ylabel("F1-score (macro)")
plt.grid(True)
plt.xticks(xticks, labels=[str(x) for x in xticks])

if not results:
    raise ValueError("Nessun file metrics trovato in Grafico_outputs")

# knee
if knee_x is not None and knee_y is not None:
    plt.axvline(x=knee_x, linestyle='--', color='red', label="Knee")

    plt.scatter(knee_x, knee_y, color='red')

    plt.text(knee_x, knee_y, f"Knee: {knee_x}", fontsize=10, ha='right', va='bottom')

plt.legend(loc="lower right")

print(f"Knee point: {knee_x} with F1-score: {knee_y}")

plot_path = os.path.join(output_dir, "f1_score_vs_sample_size.png")

plt.tight_layout()
plt.savefig(plot_path)
plt.show()