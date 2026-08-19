import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from packages.review_operations import predict_sentiment
import streamlit as st
import time


#loading the pretrained model
model = load_model('simple_rnn_imdb.h5')


 
 
## streamlit 

st.title("IMDB Movie review Sentiment Analysis")
st.write('Enter a movie review to classify its as a positive or negative')

user_input = st.text_area("Write a movie review")

button = st.button("Post")

if button:
    if user_input:

        sentiment, prediction_score = predict_sentiment(model=model, review=user_input)
        confidence = prediction_score * 100
        st.write(f"Confidence: {confidence:.1f}")
        st.info(f'{sentiment} Review')

        #progress bar 
        progress_bar = st.progress(0.0)

        ##filling animation
        for step in range(int(confidence + 1)):
            time.sleep(0.01)
            progress_bar.progress(step/ 100.0)
    else:
        st.warning('Please add a review before posting')
