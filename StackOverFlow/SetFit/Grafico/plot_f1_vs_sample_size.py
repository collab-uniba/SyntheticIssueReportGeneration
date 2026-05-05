import sys
import os
import json
from datasets import load_dataset, Dataset
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) != 3:
    sys.exit(1)

train_csv_path = sys.argv[1]
test_csv_path = sys.argv[2]

dataset = load_dataset(
    "csv",
    data_files={
        "train": train_csv_path,
        "test": test_csv_path
    },
    delimiter=";",
    quotechar='"'
)
# 5, 10, 15, 20, 25, 50, 100, 200, 
sample_sizes = ["all_capped"]

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

    capped_size = min(len(dataset["train"]), 1000)

    train_dataset = (
        dataset["train"].shuffle(seed=42).select(range(capped_size))
        if size == "all_capped"
        else sample_dataset(dataset["train"], label_column="Polarity", num_samples=size)
    )

    actual_size = capped_size if size == "all_capped" else size

    size_label = f"all_capped_{actual_size}" if size == "all_capped" else str(size)

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

df_results = pd.DataFrame(results)

df_results = df_results.sort_values("sample_size")

plt.figure(figsize=(10, 6))
plt.plot(df_results["sample_size"], df_results["f1_score"], marker='o')
plt.title("F1-score vs Sample Size")
plt.xlabel("Numero di campioni nel training set")
plt.ylabel("F1-score (macro)")
plt.grid(True)
plt.xticks(df_results["sample_size"])
plt.tight_layout()

plot_path = os.path.join(output_dir, "f1_score_vs_sample_size.png")
plt.savefig(plot_path)
plt.show()
