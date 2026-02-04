# predict_mood_transformer.py

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import pandas as pd

# Load trained model and tokenizer
from pathlib import Path

MODEL_PATH = Path("/Users/saroshnadaf/PycharmProjects/AI-Powered Mood Journal/mood_transformer_model")

tokenizer = DistilBertTokenizerFast.from_pretrained(str(MODEL_PATH), local_files_only=True)
model = DistilBertForSequenceClassification.from_pretrained(str(MODEL_PATH), local_files_only=True)
LABEL_PATH = "/Users/saroshnadaf/PycharmProjects/AI-Powered Mood Journal/label_encoder.csv"

print("🔄 Loading model and tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Load label encoder
label_df = pd.read_csv(LABEL_PATH)
id2label = dict(zip(label_df['encoded'], label_df['label']))
def predict_mood(text):
    """Predict mood from user input text or emoji."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_id = torch.argmax(outputs.logits, dim=1).item()
        return id2label[predicted_id]

# Interactive CLI
if __name__ == "__main__":
    print("\n💬 AI Mood Journal is ready! Type something to analyze your mood.")
    print("Type 'exit' to quit.\n")

    while True:
        text = input("📝 Enter text or emoji: ")
        if text.lower() == "exit":
            print("👋 Exiting Mood Journal.")
            break
        mood = predict_mood(text)
        print(f"🧠 Predicted Mood: {mood}\n")