import pandas as pd

# --- Load the GoEmotions raw CSVs ---
df1 = pd.read_csv("/Users/saroshnadaf/PycharmProjects/AI-Powered Mood Journal/datasets/goemotions/goemotions_1.csv")
df2 = pd.read_csv("/Users/saroshnadaf/PycharmProjects/AI-Powered Mood Journal/datasets/goemotions/goemotions_2.csv")
df3 = pd.read_csv("/Users/saroshnadaf/PycharmProjects/AI-Powered Mood Journal/datasets/goemotions/goemotions_3.csv")

# Combine all into one dataframe
df = pd.concat([df1, df2, df3], ignore_index=True)

print("✅ Loaded GoEmotions dataset with shape:", df.shape)
print(df.head())

# --- Define mapping 27 → 10 moods ---
mapping = {
    "sadness": "Depressed",
    "grief": "Despairing",
    "remorse": "Despairing",
    "fear": "Despairing",
    "anger": "Despairing",
    "disappointment": "Melancholic",
    "disapproval": "Melancholic",
    "disgust": "Melancholic",
    "embarrassment": "Depressed",
    "nervousness": "Depressed",
    "annoyance": "Apathetic",

    "neutral": "Neutral",
    "realization": "Apathetic",
    "confusion": "Apathetic",
    "curiosity": "Calm",
    "relief": "Calm",

    "optimism": "Content",
    "admiration": "Content",
    "approval": "Content",
    "gratitude": "Content",
    "caring": "Content",

    "joy": "Happy",
    "amusement": "Happy",
    "surprise": "Happy",

    "pride": "Elated",
    "love": "Elated",
    "desire": "Elated",
    "excitement": "Elated",
}

# Keep only text + mapped labels
emotion_columns = list(mapping.keys())

def map_to_category(row):
    for emotion in emotion_columns:
        if row.get(emotion, 0) == 1:   # if this emotion is marked
            return mapping[emotion]
    return "Neutral"

df["final_label"] = df.apply(map_to_category, axis=1)

cleaned = df[["text", "final_label"]]

# Save cleaned dataset
cleaned.to_csv("cleaned_goemotions.csv", index=False)
print("✅ Saved cleaned dataset: cleaned_goemotions.csv")
print(cleaned.head())
