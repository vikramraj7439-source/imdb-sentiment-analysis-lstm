🎬 IMDB Sentiment Analysis using LSTM
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-93%25+-brightgreen.svg)
📌 Project Overview
This project builds a deep learning model to classify movie reviews as Positive or Negative using the IMDB dataset. The model is trained using an LSTM (Long Short-Term Memory) network and achieves 93%+ accuracy on unseen test data.
> **"Can a machine understand human sentiment from raw text?"**  
> This project answers that — using deep learning and NLP.
---
🎯 Problem Statement
Given a movie review in plain text, predict whether the sentiment is:
✅ Positive — the reviewer liked the movie
❌ Negative — the reviewer disliked the movie
---
📊 Dataset
Property	Details
Source	IMDB Movie Reviews Dataset
Total Reviews	50,000
Training Set	25,000 reviews
Test Set	25,000 reviews
Classes	Binary (Positive / Negative)
Balance	Perfectly balanced (50/50)
---
🏗️ Model Architecture
```
Input Text
    ↓
Tokenization + Padding
    ↓
Embedding Layer (word vectors)
    ↓
LSTM Layer (captures sequence context)
    ↓
Dense Layer
    ↓
Output: Positive / Negative
```
Layer	Details
Embedding	Converts words to dense vectors
LSTM	Captures long-range text dependencies
Dense	Fully connected output layer
Activation	Sigmoid (binary classification)
Loss Function	Binary Crossentropy
Optimizer	Adam
---
📈 Results
Metric	Score
Test Accuracy	93%+
Loss Function	Binary Crossentropy
Training Strategy	Early stopping to prevent overfitting
---
🛠️ Tech Stack
Language: Python 3.8+
Deep Learning: TensorFlow, Keras
NLP: Keras Tokenizer, Sequence Padding
Data Processing: NumPy, Pandas
Visualization: Matplotlib
---
🚀 How to Run
1. Clone the repository
```bash
git clone https://github.com/vikramraj7439-source/imdb-sentiment-analysis-lstm.git
cd imdb-sentiment-analysis-lstm
```
2. Install dependencies
```bash
pip install tensorflow numpy pandas matplotlib
```
3. Run the notebook
```bash
jupyter notebook imdb_sentiment_analysis.ipynb
```
---
📁 Project Structure
```
imdb-sentiment-analysis-lstm/
│
├── imdb_sentiment_analysis.ipynb   # Main notebook
├── README.md                        # Project documentation
└── requirements.txt                 # Dependencies
```
---
🔍 Key Learnings
How LSTM networks capture sequential dependencies in text
NLP preprocessing: tokenization, word indexing, sequence padding
Embedding layers and how they represent words as vectors
Handling binary classification with deep learning
Preventing overfitting using early stopping
---
👨‍💻 Author
Vikram Kumar  
Final-year B.Sc. Computer Science & Machine Learning  
Loyola Academy, Hyderabad
![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)
![GitHub](https://img.shields.io/badge/GitHub-Follow-black)
---
🌟 If you found this helpful, please star the repo!
