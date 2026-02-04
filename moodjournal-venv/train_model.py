import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# --- Load cleaned data ---
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

X_train = train_df["text"]
y_train = train_df["label"]
X_test = test_df["text"]
y_test = test_df["label"]

# Remove any rows with missing or empty text
X_train = X_train.fillna("").astype(str)
X_test = X_test.fillna("").astype(str)

# (Optional) remove pure whitespace entries
train_mask = X_train.str.strip() != ""
test_mask = X_test.str.strip() != ""
X_train, y_train = X_train[train_mask], y_train[train_mask]
X_test, y_test = X_test[test_mask], y_test[test_mask]

# --- Convert text to TF-IDF features ---
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# --- Train the model ---
model = LogisticRegression(max_iter=300)
model.fit(X_train_tfidf, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test_tfidf)

print("✅ Model Performance:\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- Save model and vectorizer ---
joblib.dump(model, "mood_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\n✅ Model and vectorizer saved successfully!")

# inside train_model.py, after evaluation:
from sklearn.metrics import classification_report
import pandas as pd

report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv("tfidf_results.csv", index=True)
print("✅ Results saved to tfidf_results.csv")