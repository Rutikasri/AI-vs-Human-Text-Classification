import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download nltk resources (runs only first time)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


# Cache model loading for performance
@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer


model, vectorizer = load_model()


# Preprocessing setup
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)


# Streamlit UI
st.title("AI vs Human Text Detector")

st.write(
    "Paste a paragraph below and the model will predict whether the text "
    "was written by AI or by a human."
)


user_text = st.text_area(
    "Enter text here",
    height=200
)


if st.button("Predict"):

    if user_text.strip() == "":
        st.warning("Please enter some text.")

    elif len(user_text.split()) < 10:
        st.warning("Please enter at least 20 words for reliable prediction.")

    else:
        try:

            cleaned = clean_text(user_text)

            vector = vectorizer.transform([cleaned])

            prediction = model.predict(vector)[0]

            if prediction == 1:
                st.error("Prediction: AI Generated Text")

            else:
                st.success("Prediction: Human Written Text")

        except Exception as e:
            st.error(f"Error during prediction: {e}")