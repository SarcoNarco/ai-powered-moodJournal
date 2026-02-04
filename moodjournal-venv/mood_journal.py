from textblob import TextBlob

def analyze_mood():
    #user i/p
    user_input = input("How are you doing today")

    #sentiment analysis
    blob = TextBlob(user_input)
    sentiment = blob.sentiment.polarity

    print("\nYour Entry:", user_input)
    print("Sentiment Score:", sentiment)

    if sentiment > 0.2:
        print("🌤️️You seem to have a positive mood toady")
    elif sentiment < -0.2:
        print("You might be feeling a little down today☁️")
    else:
        print("Well, everything seems normal🌫️")

if __name__ == "__main__":
    analyze_mood()
