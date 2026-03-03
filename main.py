import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN
from tensorflow.keras.preprocessing.sequence import pad_sequences

# CONFIG
max_len = 500
num_words = 10000


# LOAD MODEL & TOKENIZER
@st.cache_resource
def load_resources():
    # Rebuild architecture
    model = Sequential()
    model.add(Embedding(num_words, 128, input_length=max_len)) ## Embedding layer
    model.add(SimpleRNN(128, activation="relu")) ## Simple RNN layer
    model.add(Dense(1, activation="sigmoid")) ## Output layer
    model.build(input_shape=(None, max_len))
    # Load trained weights
    model.load_weights("model.weights.h5")
    # Load tokenizer
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_resources()

# Prediction Function
def predict_sentiment(text):
    sequence_input = tokenizer.texts_to_sequences([text])
    padded_input = pad_sequences(sequence_input, maxlen=max_len)
    prediction = model.predict(padded_input, verbose=0)
    score = float(prediction[0][0])
    sentiment = "Positive" if score >= 0.5 else "Negative"
    return sentiment, score

# Streamlit UI
st.title("IMDB Movie Review Sentiment Analysis (LSTM)")
st.write("Enter a movie review to classify it as Positive or Negative")

user_input = st.text_area("Enter your movie review here:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        sentiment, score = predict_sentiment(user_input)
        st.success(f"Sentiment: {sentiment}")
        st.write(f"Confidence Score: {score:.4f}")