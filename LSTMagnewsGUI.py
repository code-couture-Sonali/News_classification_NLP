import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import pickle
import streamlit as st

# Load the pre-trained LSTM model for AG News classification
model = load_model('lstm_news_classification.h5')

# Load the saved tokenizer
with open('news_tokenizer_lstm.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Class labels mapping
class_labels = ['World', 'Sports', 'Business', 'Sci/Tech']

# Function to preprocess user input text
def preprocess_text(text):
    # Convert text to sequence of integers
    sequence_text = tokenizer.texts_to_sequences([text])
    # Pad the sequence to max length 100 (consistent with training)
    padded_text = sequence.pad_sequences(sequence_text, maxlen=100)
    return padded_text

# Function to predict class and confidence
def predict_news_class(text):
    preprocessed_text = preprocess_text(text)
    prediction = model.predict(preprocessed_text)
    class_id = np.argmax(prediction[0])
    predicted_class = class_labels[class_id]
    confidence = prediction[0][class_id]
    return predicted_class, confidence, prediction[0]

# Streamlit app UI
st.title('AG News Topic Classification using LSTM')
st.write('Enter a news title or description to classify its topic.')

# User input
user_input = st.text_area('News Title or Description')

if st.button('Classify'):
    if user_input:
        pred_class, conf_score, probs = predict_news_class(user_input)
        st.write(f'**Predicted Topic:** {pred_class}')
        st.write(f'**Confidence Score:** {conf_score:.4f}')
        st.write('---')
        st.write('**Probability Distribution:**')
        for i, label in enumerate(class_labels):
            st.write(f'{label}: {probs[i]:.4f}')
    else:
        st.write('Please enter some text to classify.')
