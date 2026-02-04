import pandas as pd

# Load the dataset
df = pd.read_csv("/Users/saroshnadaf/PycharmProjects/AI-Powered Mood Journal/emoji/emoji_sentiment_dataset.csv")

print("✅ Loaded Emoji Sentiment dataset with shape:", df.shape)
print(df.head())

# Function to map sentiment_score (diff) to your 10 categories
def map_sentiment_to_mood(score):
    if score <= -0.6:
        return "Despairing"
    elif -0.6 < score <= -0.3:
        return "Depressed"
    elif -0.3 < score < 0.0:
        return "Melancholic"
    elif score == 0.0:
        return "Neutral"
    elif 0.0 < score <= 0.2:
        return "Apathetic"
    elif 0.2 < score <= 0.4:
        return "Calm"
    elif 0.4 < score <= 0.6:
        return "Content"
    elif 0.6 < score <= 0.75:
        return "Happy"
    elif 0.75 < score <= 0.9:
        return "Elated"
    else:
        return "Ecstatic"

# Apply mapping using the 'diff' column
df["final_label"] = df["diff"].apply(map_sentiment_to_mood)

# Keep only emoji + final label
cleaned = df[["emoji", "final_label"]].rename(columns={"emoji": "text"})

# Save cleaned dataset
cleaned.to_csv("cleaned_emoji.csv", index=False)

print("✅ Saved cleaned dataset: cleaned_emoji.csv")
print(cleaned.head())