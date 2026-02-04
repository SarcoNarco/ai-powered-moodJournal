from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

MODEL_PATH = "/Users/saroshnadaf/PycharmProjects/AI-Powered Mood Journal/mood_transformer_model"
print("🔄 Loading model and tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
print("✅ Model loaded successfully!")