from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb

word_index = imdb.get_word_index()
reversed_word_index = {value:key for key, value in word_index.items()}

def decode_review(encoded_review):
    return ' '.join([reversed_word_index.get(i-3, '?') for i in encoded_review])
VOCAB_SIZE = 10000
## function to preprocess te input 
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [
        word_index[word] + 3 if (word in word_index and (word_index[word] + 3) < VOCAB_SIZE) else 2
        for word in words
    ]
    padded_review = pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(model, review):
    padded_input = preprocess_text(review)
    prediction = model.predict(padded_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negetive'
    return sentiment, prediction[0][0]