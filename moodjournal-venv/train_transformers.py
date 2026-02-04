import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
import torch

# --- Load data ---
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

# --- Clean and ensure text is string ---
train_df["text"] = train_df["text"].fillna("").astype(str)
test_df["text"] = test_df["text"].fillna("").astype(str)

# --- Encode labels numerically ---
le = LabelEncoder()
train_df["label_id"] = le.fit_transform(train_df["label"])
test_df["label_id"] = le.transform(test_df["label"])

# Create Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df[["text", "label_id"]])
test_dataset = Dataset.from_pandas(test_df[["text", "label_id"]])

# Rename label_id → labels (required by Trainer)
train_dataset = train_dataset.rename_column("label_id", "labels")
test_dataset = test_dataset.rename_column("label_id", "labels")

# --- Tokenizer ---
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# --- Model ---
num_labels = len(le.classes_)
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

args = TrainingArguments(
    output_dir="bert_results",
    do_eval=True,                 # enable evaluation
    save_total_limit=2,           # keep only last two checkpoints
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="logs"
)

# --- Trainer ---
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=None,  # or a custom function if you want accuracy later
)

# --- Train ---
trainer.train()

# --- Evaluate ---
metrics = trainer.evaluate()
print("✅ Evaluation metrics:", metrics)

# --- Save model & label encoder ---
model.save_pretrained("mood_transformer_model")
tokenizer.save_pretrained("mood_transformer_model")
pd.Series(le.classes_).to_csv("label_encoder.csv", index=False)
print("✅ Model + tokenizer + label encoder saved!")

