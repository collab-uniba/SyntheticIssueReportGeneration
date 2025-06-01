import argparse
from datasets import load_dataset, DatasetDict
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from sklearn.model_selection import train_test_split
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--train_file", type=str, required=True, help="Percorso al file CSV di training.")
parser.add_argument("-t", "--test_file", type=str, default=None, help="Percorso al file CSV di test (opzionale).")
parser.add_argument("-n", "--num_samples", type=int, default=20, help="Numero di sample per etichetta da usare per il training. 0 = tutto il dataset.")
parser.add_argument("-s", "--split_ratio", type=float, default=0.3, help="Percentuale del dataset di training da usare come test se non è fornito un test set.")

args = parser.parse_args()

data_files = {"train": args.train_file}
if args.test_file:
    data_files["test"] = args.test_file

dataset = load_dataset(
    "csv",
    data_files=data_files,
    delimiter=";",
    quotechar='"'
)

# Se non c'è test set, fai uno split del train (caso del dataset di github gold)
if "test" not in dataset:
    df = dataset["train"].to_pandas()
    df_train, df_test = train_test_split(df, test_size=args.split_ratio, stratify=df["Polarity"], random_state=42)
    dataset = DatasetDict({
        "train": dataset["train"].from_pandas(df_train.reset_index(drop=True)),
        "test": dataset["train"].from_pandas(df_test.reset_index(drop=True))
    })

# per usare tutto il dataset e non fare un sample passare come parametro n<=0
if args.num_samples > 0:
    train_dataset = sample_dataset(dataset["train"], label_column="Polarity", num_samples=args.num_samples)
else:
    train_dataset = dataset["train"]

test_dataset = dataset["test"]

model = SetFitModel.from_pretrained(
    "all-mpnet-base-v2",
    labels=["negative", "positive", "neutral"],
)

training_args = TrainingArguments(
    batch_size=16,
    num_epochs=4,
    eval_strategy="no",
    save_strategy="no",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    metric="accuracy",
    column_mapping={"Text": "text", "Polarity": "label"}
)

trainer.train()

metrics = trainer.evaluate(test_dataset)
print(metrics)

test_ids = test_dataset["ID"]
test_texts = test_dataset["Text"]
predicted_labels = model.predict(test_texts)

df_results = pd.DataFrame({
    "ID": test_ids,
    "Text": test_texts,
    "Prediction": predicted_labels
})

df_results.to_csv("test_predictions.csv", index=False, encoding="utf-8")
