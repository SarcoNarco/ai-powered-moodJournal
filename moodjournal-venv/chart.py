# =============================================
# DistilBERT Emotion Classifier Report & Chart
# =============================================

from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Step 1: Input your true and predicted labels ----
# Replace with your actual labels
y_true = ['Content', 'Happy', 'Neutral', 'Depressed', 'Calm', 'Neutral', 'Happy', 'Despairing', 'Content', 'Melancholic']
y_pred = ['Content', 'Happy', 'Neutral', 'Despairing', 'Content', 'Neutral', 'Happy', 'Depressed', 'Content', 'Melancholic']

# ---- Step 2: Generate classification report ----
report = classification_report(y_true, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()

# ---- Step 3: Clean the report (keep only emotion rows) ----
df_emotions = df_report.drop(['accuracy', 'macro avg', 'weighted avg'])

# ---- Step 4: Plot Precision, Recall, and F1-score ----
sns.set(style='whitegrid')
colors = ['#66c2a5', '#fc8d62', '#8da0cb']

plt.figure(figsize=(12, 6))
df_emotions[['precision', 'recall', 'f1-score']].plot(
    kind='bar',
    color=colors,
    edgecolor='black',
    figsize=(12, 6)
)

plt.title('DistilBERT Emotion Classification Report', fontsize=14, fontweight='bold')
plt.xlabel('Emotions', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.ylim(0, 1)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Metrics', loc='lower right')
plt.tight_layout()
plt.show()