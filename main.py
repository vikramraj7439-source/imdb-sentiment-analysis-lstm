import numpy as np
import streamlit as st
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# -----------------------
# Load Model & Tokenizer
# -----------------------

model = load_model("sentiment_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 500


# -----------------------
# Prediction Function
# -----------------------

def predict_sentiment(text):

    sequence_input = tokenizer.texts_to_sequences([text])
    padded_input = pad_sequences(sequence_input, maxlen=max_len)

    prediction = model.predict(padded_input)

    sentiment = "Positive" if prediction[0][0] >= 0.5 else "Negative"

    return sentiment, float(prediction[0][0])


# -----------------------
# Streamlit App
# -----------------------

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as Positive or Negative")

user_input = st.text_area("Enter your movie review here:")

if st.button("Classify"):

    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        sentiment, score = predict_sentiment(user_input)

        st.write(f"Sentiment: {sentiment}")
        st.write(f"Prediction Score: {score:.4f}")