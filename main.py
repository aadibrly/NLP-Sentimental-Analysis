import time
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from packages.review_operations import predict_sentiment

# Page setup
st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for review history
if "history" not in st.session_state:
    st.session_state.history = []

# Cache model loading so it doesn't reload on every interaction
@st.cache_resource
def load_rnn_model():
    return load_model('simple_rnn_imdb.h5')

model = load_rnn_model()

#SIDEBAR
with st.sidebar:
    st.title("🎬 IMDb Sentiment Analyzer")
    st.markdown("---")
    
    st.subheader("🧭 Navigation")
    st.markdown("""
    - [🏠 Home](#imdb-movie-review-sentiment-analysis)
    - [📊 Dashboard](#analysis-result)
    - [🕒 History](#recent-reviews)
    """)
    
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.caption(
        "This app uses a **SimpleRNN** deep learning model trained on IMDb movie reviews "
        "to predict whether an input review is **Positive** or **Negative**."
    )
    
    st.markdown("---")
    st.subheader("⚙️ Model Info")
    st.markdown("""
    - **Model:** SimpleRNN
    - **Dataset:** IMDb Reviews (10,000 Vocab)
    - **Task:** Binary Classification
    - **Framework:** TensorFlow / Keras
    """)
    
    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit")

#MAIN SECTION
st.title("IMDb Movie review Sentiment Analysis")
st.write("Enter a movie review to classify it as **Positive** or **Negative**.")

# Input Layout (Text area on left, Tips on right)
col_input, col_tips = st.columns([3, 1], gap="medium")

with col_input:
    st.subheader("✍️ Write a movie review")
    user_input = st.text_area(
        label="Write a movie review",
        placeholder="e.g., The cinematography was breathtaking and the acting was top notch!",
        height=130,
        label_visibility="collapsed"
    )
    button = st.button("Post", type="primary")

with col_tips:
    with st.container(border=True):
        st.markdown("💡 **Tips**")
        st.caption("• Write your review in English\n• Longer reviews give better predictions\n• Be honest and expressive!")

#ANALYSIS RESULT
st.markdown("---")
st.markdown('<a name="analysis-result"></a>', unsafe_allow_html=True)
st.subheader("🔍 Analysis Result")

if button:
    if user_input.strip():
        with st.spinner("Analyzing sentiment..."):
            sentiment, prediction_score = predict_sentiment(model=model, review=user_input)
            
            # True confidence score calculation
            display_score = prediction_score if sentiment == "Positive" else (1.0 - prediction_score)
            confidence = display_score * 100

            # Store in session state history (newest first)
            st.session_state.history.insert(0, {
                "review": user_input,
                "sentiment": sentiment,
                "confidence": f"{confidence:.1f}%",
                "score_raw": display_score
            })

    else:
        st.warning("Please add a review before posting")

# Display Result Card if history exists
if st.session_state.history:
    latest = st.session_state.history[0]
    is_positive = latest["sentiment"] == "Positive"
    
    with st.container(border=True):
        col_icon, col_sentiment, col_score, col_meter = st.columns([1, 2, 2, 3], vertical_alignment="center")
        
        # 1. Smiley Emoji
        with col_icon:
            if is_positive:
                st.markdown("<div style='font-size: 50px; text-align: center;'>🟢 😊</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='font-size: 50px; text-align: center;'>🔴 😞</div>", unsafe_allow_html=True)
        
        # 2. Sentiment Label
        with col_sentiment:
            st.markdown(f"### {latest['sentiment']} Review")
            st.caption(f"Predicted class: **{latest['sentiment'].upper()}**")
        
        # 3. Confidence Text
        with col_score:
            st.caption("Confidence Score")
            st.markdown(f"## {latest['confidence']}")
            if float(latest['score_raw']) >= 0.75:
                st.caption("High Confidence ⭐")
            else:
                st.caption("Moderate Confidence")
                
        # 4. Animated Confidence Progress Meter
        with col_meter:
            st.caption("Confidence Meter")
            progress_bar = st.progress(0.0)
            target_int = int(float(latest['score_raw']) * 100)
            for step in range(target_int + 1):
                time.sleep(0.005)
                progress_bar.progress(step / 100.0)
else:
    st.info("Submit a review above to see the sentiment analysis result.")

# recent review history
st.markdown("---")
st.markdown('<a name="recent-reviews"></a>', unsafe_allow_html=True)
st.subheader("🕒 Recent Reviews")

if st.session_state.history:
    # Table Header
    h1, h2, h3 = st.columns([6, 2, 2])
    h1.markdown("**Review**")
    h2.markdown("**Prediction**")
    h3.markdown("**Confidence**")
    st.divider()
    
    # List each review in history
    for item in st.session_state.history:
        r1, r2, r3 = st.columns([6, 2, 2])
        r1.write(f"\"{item['review']}\"")
        
        if item["sentiment"] == "Positive":
            r2.markdown("🟢 **Positive**")
        else:
            r2.markdown("🔴 **Negative**")
            
        r3.write(item["confidence"])
        st.divider()
else:
    st.caption("No reviews submitted yet in this session.")