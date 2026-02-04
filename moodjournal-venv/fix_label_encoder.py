import pandas as pd

# Load your single-column CSV
df = pd.read_csv("label_encoder.csv")

# Rename the column to something clear
df.columns = ["label"]

# Create numeric IDs (0 → n-1)
df["encoded"] = range(len(df))

# Save back properly
df.to_csv("label_encoder.csv", index=False)

print("✅ Fixed label_encoder.csv")
print(df.head())