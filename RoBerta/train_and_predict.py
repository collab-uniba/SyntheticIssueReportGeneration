from datasets import load_dataset, Features, Value
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import torch
from evaluate import load as load_metric
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-d','--train_file', type=str, required=True, help='Path al file CSV di training')
parser.add_argument('-t','--test_file', type=str, required=True, help='Path al file CSV di test')
parser.add_argument('--output_dir', type=str, default='.', help='Directory dove salvare i risultati')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# Definisci le feature per forzare tutti i tipi
features = Features({
    "ID": Value("string"),
    "Polarity": Value("string"),
    "Text": Value("string")
})

dataset = load_dataset(
    'csv',
    data_files={'train': args.train_file, 'test': args.test_file},
    delimiter=';',
    quotechar='"',
    features=features
)

train_dataset = dataset['train']
test_dataset = dataset['test']


# Encode "Polarity" labels into integers
label_encoder = LabelEncoder()
label_encoder.fit(train_dataset['Polarity'])
train_dataset = train_dataset.map(lambda x: {'label': label_encoder.transform([x['Polarity']])[0]})
test_dataset = test_dataset.map(lambda x: {'label': label_encoder.transform([x['Polarity']])[0]})

model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(example):
    return tokenizer(example['Text'], truncation=True)

tokenized_train  = train_dataset.map(preprocess_function, batched=True)
tokenized_test  = test_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
tokenized_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

load_accuracy  = load_metric("accuracy")
load_f1  = load_metric("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy .compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1 .compute(predictions=predictions, references=labels, average="weighted")["f1"]
    return {"accuracy": accuracy, "f1": f1}

training_args = TrainingArguments(
    output_dir=args.output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=16,#
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    save_strategy="epoch",
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

predictions_output = trainer.predict(tokenized_test)
pred_labels = np.argmax(predictions_output.predictions, axis=1)

predicted_labels_text = label_encoder.inverse_transform(pred_labels)

test_df = test_dataset.to_pandas()

test_df["Prediction"] = predicted_labels_text

columns_to_save = ["ID","Prediction","Text"]  # specifica le colonne da salvare
columns_to_save = [col for col in columns_to_save if col in test_df.columns]  # filtra solo le colonne esistenti

predictions_path = os.path.join(args.output_dir, "test_predictions.csv")
test_df[columns_to_save].to_csv(predictions_path, index=False)

 