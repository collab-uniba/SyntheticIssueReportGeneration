import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

if len(sys.argv) != 2:
    sys.exit(1)

train_csv_path = sys.argv[1]

df_train = pd.read_csv(train_csv_path, sep=";", quotechar='"')

train_df, internal_test_df = train_test_split(
    df_train,
    test_size=2/3,
    stratify=df_train["Polarity"],
    random_state=42
)

train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
internal_test_dataset = Dataset.from_pandas(internal_test_df.reset_index(drop=True))

sample_sizes = [5, 10, 15, 20, 25, 30, 50, 100, 150, 200,250,300, "all"]
results = []

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(output_dir, exist_ok=True)

for size in sample_sizes:
    print(f"\n===> Training con {size} esempi per classe")

    size_label = str(size) if size != "all" else f"all_{len(train_dataset)}"
    current_train = train_dataset if size == "all" else sample_dataset(train_dataset, label_column="Polarity", num_samples=size)

    model = SetFitModel.from_pretrained(
        "all-mpnet-base-v2",
        labels=["negative", "positive", "neutral"]
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
        train_dataset=current_train,
        eval_dataset=internal_test_dataset,
        metric="accuracy",
        column_mapping={"Text": "text", "Polarity": "label"}
    )

    trainer.train()

    test_texts = internal_test_dataset["Text"]
    test_labels = internal_test_dataset["Polarity"]
    predicted_labels = model.predict(test_texts)

    f1 = f1_score(test_labels, predicted_labels, average="macro")
    acc = accuracy_score(test_labels, predicted_labels)
    precision = precision_score(test_labels, predicted_labels, average="macro", zero_division=0)
    recall = recall_score(test_labels, predicted_labels, average="macro", zero_division=0)

    metrics = {
        "sample_size": size if size != "all" else len(train_dataset),
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
plt.title("F1-score vs Numero di Campioni")
plt.xlabel("Numero di campioni nel training set (per classe)")
plt.ylabel("F1-score (macro)")
plt.grid(True)
plt.xticks(df_results["sample_size"])
plt.tight_layout()

plot_path = os.path.join(output_dir, "f1_score_vs_sample_size.png")
plt.savefig(plot_path)
plt.show()
