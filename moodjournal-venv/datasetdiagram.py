# ===========================================
# Emotion Distribution Visualization
# ===========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Step 1: Load or create your dataset ----
# Replace this with your actual data
# Example: each row is one text sample and its emotion label
data = pd.DataFrame({
    'emotion': [
        'Content', 'Happy', 'Neutral', 'Depressed', 'Calm',
        'Neutral', 'Happy', 'Despairing', 'Content', 'Melancholic',
        'Happy', 'Neutral', 'Depressed', 'Content', 'Content',
        'Calm', 'Happy', 'Neutral', 'Despairing', 'Content'
    ]
})

# ---- Step 2: Count emotion frequencies ----
emotion_counts = data['emotion'].value_counts().sort_values(ascending=False)

# ---- Step 3: Bar Chart ----
plt.figure(figsize=(10,6))
sns.set(style='whitegrid', palette='muted')

sns.barplot(x=emotion_counts.index, y=emotion_counts.values, edgecolor='black')

plt.title('Emotion Distribution in Dataset', fontsize=14, fontweight='bold')
plt.xlabel('Emotion Labels', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ---- Step 4: Pie Chart ----
plt.figure(figsize=(7,7))
plt.pie(emotion_counts.values, labels=emotion_counts.index,
        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
plt.title('Emotion Distribution (Percentage)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()