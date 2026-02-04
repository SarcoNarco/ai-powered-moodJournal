
import pandas as pd
import emoji
import re
# Manual English stopwords list (no download needed)
stop_words = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up',
    'down','in','out','on','off','over','under','again','further','then',
    'once','here','there','when','where','why','how','all','any','both','each',
    'few','more','most','other','some','such','no','nor','not','only','own',
    'same','so','than','too','very','s','t','can','will','just','don','should',
    'now'
}
# Load dataset
df = pd.read_csv("cleaned_mood_dataset.csv")


# --- Preprocessing function ---
def preprocess_text(text):
    text = str(text).lower()                      # lowercase
    text = emoji.demojize(text)                   # convert 🙂 -> :slightly_smiling_face:
    text = re.sub(r"http\S+", "", text)           # remove URLs
    text = re.sub(r"[^a-zA-Z\s:]", "", text)      # remove punctuation, keep words + emoji codes
    words = text.split()                          # tokenize
    words = [w for w in words if w not in stop_words]  # remove stopwords
    return " ".join(words)

# Apply preprocessing
df["clean_text"] = df["text"].apply(preprocess_text)

# Split into train/test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["final_label"], test_size=0.2, random_state=42
)

# Save splits for training later
train_df = pd.DataFrame({"text": X_train, "label": y_train})
test_df = pd.DataFrame({"text": X_test, "label": y_test})

train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

print("✅ Preprocessing done! Saved train_data.csv and test_data.csv")
print(train_df.head())