import sys
import os
import json
from datasets import load_dataset
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

sample_sizes = [5, 10, 15, 20, 25, 30, 50, 100, 150, 200, "all"]

test_dataset = dataset["test"]
test_texts = test_dataset["Text"]
test_labels = test_dataset["Polarity"]
test_ids = test_dataset["ID"]

results = []

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(output_dir, exist_ok=True)

for size in sample_sizes:
    print(f"\n===> Training with {size} samples")

    size_label = str(size) if size != "all" else f"all_{len(dataset['train'])}"
    train_dataset = dataset["train"] if size == "all" else sample_dataset(dataset["train"], label_column="Polarity", num_samples=size)

    model = SetFitModel.from_pretrained(
        "all-mpnet-base-v2",
        labels=["negative", "positive", "neutral"],
    )

    args = TrainingArguments(
        batch_size=16,
        num_epochs=4,
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=True,
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
        "sample_size": size if size != "all" else len(dataset["train"]),
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
