# mood_journal_gui.py

import tkinter as tk
from tkinter import messagebox
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import pandas as pd
from datetime import datetime, timedelta
import os

# --- Load model and label encoder ---
MODEL_PATH = "/Users/saroshnadaf/PycharmProjects/AI-Powered Mood Journal/mood_transformer_model"
LABEL_PATH = "/Users/saroshnadaf/PycharmProjects/AI-Powered Mood Journal/label_encoder.csv"
LOG_FILE = "/Users/saroshnadaf/PycharmProjects/AI-Powered Mood Journal/data/mood_log.csv"

print("🔄 Loading model and tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH, local_files_only=True)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)

# --- Device setup ---
import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # for Mac M1/M2
model.to(device)
model.eval()

# Load label encoder
label_df = pd.read_csv(LABEL_PATH)
if "encoded" in label_df.columns:
    id2label = dict(zip(label_df["encoded"], label_df["label"]))
else:
    id2label = dict(zip(range(len(label_df)), label_df.iloc[:, 0]))


# --- Helper: Predict Mood ---
def predict_mood(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_id = torch.argmax(outputs.logits, dim=1).item()
        return id2label[predicted_id]


# --- Helper: Log Entry ---
def log_mood(entry_text, mood):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = pd.DataFrame([[now, entry_text, mood]], columns=["timestamp", "text", "mood"])

    # Ensure data directory exists
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    if os.path.exists(LOG_FILE):
        old = pd.read_csv(LOG_FILE)
        updated = pd.concat([old, new_entry], ignore_index=True)
    else:
        updated = new_entry

    updated.to_csv(LOG_FILE, index=False)
    print(f"✅ Logged mood: {mood}")


# --- Helper: Calculate Weekly Mood Score ---
def calculate_weekly_mood():
    if not os.path.exists(LOG_FILE):
        return "No logs yet.", "⚪️ No data yet. Try journaling for a few days!"

    df = pd.read_csv(LOG_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    one_week_ago = datetime.now() - timedelta(days=7)
    week_data = df[df["timestamp"] >= one_week_ago]

    if week_data.empty:
        return "No data for this week.", "⚪️ Keep journaling to see your mood patterns!"

    # Assign numeric scores for moods
    mood_scores = {
        "Ecstatic": 10, "Elated": 9, "Happy": 8,
        "Content": 7, "Calm": 6, "Neutral": 5,
        "Apathetic": 4, "Melancholic": 3,
        "Depressed": 2, "Despairing": 1
    }

    week_data["score"] = week_data["mood"].map(mood_scores).fillna(5)
    avg_score = week_data["score"].mean()

    # Decide feedback
    if avg_score >= 7.5:
        feedback = "🌞 You’ve had a bright, positive week! Keep the good vibes going!"
    elif avg_score >= 5:
        feedback = "😌 You’ve had a balanced week. Remember to take time to rest and recharge."
    else:
        feedback = "💪 It’s been a tough week, but you’re still standing. Better days are coming."

    summary = f"Average Mood Score (last 7 days): {avg_score:.2f}"
    return summary, feedback


# --- GUI Setup ---
root = tk.Tk()
root.title("AI-Powered Mood Journal")
root.geometry("500x450")
root.config(bg="#e0f7fa")

# Title
title = tk.Label(root, text="💬 AI Mood Journal", font=("Helvetica", 18, "bold"), bg="#e0f7fa", fg="#006064")
title.pack(pady=10)

# Text Entry
entry_label = tk.Label(root, text="How are you feeling today?", bg="#e0f7fa", fg="#004d40")
entry_label.pack(pady=5)

entry_box = tk.Text(root, height=5, width=50, font=("Helvetica", 12))
entry_box.pack(pady=5)


# Result Label
result_label = tk.Label(root, text="", font=("Helvetica", 14, "bold"), bg="#e0f7fa", fg="#00796b")
result_label.pack(pady=10)


# --- Functions for Buttons ---
def analyze_mood():
    text = entry_box.get("1.0", "end-1c").strip()
    if not text:
        messagebox.showwarning("Input Required", "Please enter your thoughts or mood.")
        return
    mood = predict_mood(text)
    log_mood(text, mood)
    result_label.config(text=f"🧠 Your mood seems to be: {mood}")
    entry_box.delete("1.0", "end")


def show_weekly_summary():
    summary, feedback = calculate_weekly_mood()
    messagebox.showinfo("Weekly Mood Summary", f"{summary}\n\n{feedback}")


# --- Buttons ---
# --- Buttons Frame ---
button_frame = tk.Frame(root, bg="#e0f7fa")
button_frame.pack(pady=20)

# Analyze Mood Button
analyze_btn = tk.Button(
    button_frame,
    text="🧠 Analyze Mood",
    command=analyze_mood,
    bg="#00796b",
    fg="white",
    font=("Helvetica", 13, "bold"),
    width=18,
    height=2,
    relief="raised",
    bd=3
)
analyze_btn.grid(row=0, column=0, padx=10, pady=10)

# Weekly Summary Button
summary_btn = tk.Button(
    button_frame,
    text="📅 View Weekly Summary",
    command=show_weekly_summary,
    bg="#004d40",
    fg="white",
    font=("Helvetica", 13, "bold"),
    width=22,
    height=2,
    relief="raised",
    bd=3
)
summary_btn.grid(row=0, column=1, padx=10, pady=10)

# --- Start App ---
root.mainloop()