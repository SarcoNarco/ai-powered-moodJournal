import joblib
import emoji

# Load saved model and vectorizer
model = joblib.load("mood_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def preprocess_input(text):
    # Convert emojis → text form
    try:
        text = emoji.demojize(text)
    except AttributeError:
        text = emoji.replace_emoji(text, replace=lambda ch, data_dict: f":{data_dict['en']}:" if data_dict and 'en' in data_dict else '')
    return text.lower()

while True:
    user_input = input("\n💬 Enter your mood or sentence (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break

    clean_text = preprocess_input(user_input)
    text_vector = vectorizer.transform([clean_text])
    prediction = model.predict(text_vector)[0]

    print(f"🧠 Predicted Mood: {prediction}")