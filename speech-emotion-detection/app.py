import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import librosa.display
import os

# Load pre-trained model and encoder
MODEL_PATH = "speech_emotion_detection_model.h5"
model = load_model(MODEL_PATH)

# Emotion categories
emotions = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
encoder = OneHotEncoder(categories=[emotions], sparse_output=False)
encoder.fit(np.array(emotions).reshape(-1, 1))

def extract_features(file_path):
    """Extract MFCC features from an audio file."""
    y, sr = librosa.load(file_path, sr=None, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

def predict_emotion(file_path):
    """Predict emotion from audio file."""
    features = extract_features(file_path)
    features = features[np.newaxis, ..., np.newaxis]  # Reshape for model
    prediction = model.predict(features)
    
    # Convert the predicted class index to a one-hot encoded array
    predicted_one_hot = np.zeros((1, len(emotions)))
    predicted_one_hot[0, np.argmax(prediction)] = 1
    
    # Use inverse_transform to get the label
    predicted_emotion = encoder.inverse_transform(predicted_one_hot)
    return predicted_emotion[0][0], np.max(prediction)


def display_waveform(file_path):
    """Display waveform of the audio file."""
    y, sr = librosa.load(file_path, sr=None)
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, alpha=0.8)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    st.pyplot(plt.gcf())

# Streamlit App
st.set_page_config(page_title="Speech Emotion Detection", layout="centered")
st.title("🎙️ Speech Emotion Detection")
st.markdown("Detect emotions from speech audio files with a cutting-edge LSTM model!")

st.sidebar.title("Upload an audio file")
# st.sidebar.markdown("Upload an audio file below:")

audio_file = st.sidebar.file_uploader("", type=["wav", "mp3"])

if audio_file is not None:
    # Save uploaded audio to disk
    file_path = os.path.join("temp_audio", audio_file.name)
    os.makedirs("temp_audio", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(audio_file.getbuffer())

    st.audio(file_path, format="audio/wav")

    # Display waveform
    st.markdown("### Waveform")
    display_waveform(file_path)

    # Predict emotion
    st.markdown("### Prediction")
    with st.spinner("Analyzing the audio..."):
        predicted_emotion, confidence = predict_emotion(file_path)
    
    st.markdown(f"**Predicted Emotion:** :red[{predicted_emotion}]")
    st.markdown(f"**Confidence:** {confidence*100:.2f}%")

    # Cleanup temporary file
    os.remove(file_path)

st.sidebar.markdown("---")
st.sidebar.markdown("**About**: This app uses a deep learning LSTM model to detect emotions in speech audio files. Developed by Aditya Gujar and team.")
