import pandas as pd

# Load cleaned GoEmotions
go_df = pd.read_csv("cleaned_goemotions.csv")

# Load cleaned Emoji dataset
emoji_df = pd.read_csv("cleaned_emoji.csv")

# Merge them
merged = pd.concat([go_df, emoji_df], ignore_index=True)

print("✅ Combined dataset shape:", merged.shape)
print(merged.head())

# Save final dataset
merged.to_csv("cleaned_mood_dataset.csv", index=False)
print("✅ Saved merged dataset: cleaned_mood_dataset.csv")