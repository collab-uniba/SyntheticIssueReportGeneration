import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import torch

# Verifica se CUDA è disponibile e stampa informazioni sulla GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device in uso:", device)

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

sample_sizes = [5, 10, 15, 20, 25, 50, 100, 200, "all_capped"]
results = []

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(output_dir, exist_ok=True)

for size in sample_sizes:
    print(f"\n===> Training con {size} esempi per classe")
    
    capped_size = min(len(train_dataset), 1000)

    current_train = (
        train_dataset.shuffle(seed=42).select(range(capped_size))
        if size == "all_capped"
        else sample_dataset(train_dataset, label_column="Polarity", num_samples=size)
    )

    actual_size = capped_size if size == "all_capped" else size

    size_label = f"all_capped_{actual_size}" if size == "all_capped" else str(size) 

    model = SetFitModel.from_pretrained(
        "all-mpnet-base-v2",
        labels=["negative", "positive", "neutral"]
    )

    args = TrainingArguments(
        batch_size=16,
        num_epochs=4,
        evaluation_strategy="no",
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

# trova plateau
epsilon = 0.015

df_results["delta_f1"] = df_results["f1_score"].diff()

plateau_points = df_results[df_results["delta_f1"] < epsilon]

plateau_x = None
plateau_y = None

if len(plateau_points) > 0:
    first_plateau = plateau_points.iloc[0]
    plateau_x = int(first_plateau["sample_size"])
    plateau_y = first_plateau["f1_score"]

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

# plateau
if plateau_x is not None and plateau_y is not None:
    plt.axvline(x=plateau_x, linestyle='--', color='green', label="Plateau")

    plt.scatter(plateau_x, plateau_y, color='green')

    plt.text(plateau_x, plateau_y, f"Plateau: {plateau_x}", fontsize=10, ha='left', va='top')

plt.legend(loc="lower right")

print(f"Knee point: {knee_x} with F1-score: {knee_y}")
print(f"Plateau point: {plateau_x} with F1-score: {plateau_y}")

plot_path = os.path.join(output_dir, "f1_score_vs_sample_size.png")

plt.tight_layout()
plt.savefig(plot_path)
plt.show()