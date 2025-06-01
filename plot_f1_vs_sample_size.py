from datasets import load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from sklearn.metrics import f1_score
import pandas as pd
import matplotlib.pyplot as plt

##per colab
#import os
#os.environ["WANDB_DISABLED"] = "true"
##

dataset = load_dataset(
    "csv",
    data_files={
        "train": "..\\train_StackOverFlow.csv",
        "test": "..\\test_StackOverFlow.csv"
    },
    delimiter=";",
    quotechar='"'
)

sample_sizes = [5, 10, 15, 20, 25, 30, 50, 100, 150, 200, "all"]

test_dataset = dataset["test"]
test_texts = test_dataset["Text"]
test_labels = test_dataset["Polarity"]

results = []

for size in sample_sizes:
    print(f"\n===> Training with {size} samples")

    if size == "all":
        train_dataset = dataset["train"]
    else:
        train_dataset = sample_dataset(dataset["train"], label_column="Polarity", num_samples=size)

    model = SetFitModel.from_pretrained(
        "all-mpnet-base-v2",
        labels=["negative", "positive", "neutral"],
    )

    args = TrainingArguments(
        batch_size=16,
        num_epochs=4,
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=None,
        metric="accuracy",
        column_mapping={"Text": "text", "Polarity": "label"}
    )

    trainer.train()

    predicted_labels = model.predict(test_texts)

    f1 = f1_score(test_labels, predicted_labels, average="macro")
    results.append({"sample_size": size if size != "all" else len(dataset["train"]), "f1_score": f1})
    print(f"F1-score: {f1:.4f}")

df_results = pd.DataFrame(results)


plt.figure(figsize=(10, 6))
plt.plot(df_results["sample_size"], df_results["f1_score"], marker='o')
plt.title("F1-score vs Sample Size")
plt.xlabel("Numero di campioni nel training set")
plt.ylabel("F1-score (macro)")
plt.grid(True)
plt.xticks(df_results["sample_size"])
plt.tight_layout()
plt.savefig("f1_score_vs_sample_size.png")
plt.show()
