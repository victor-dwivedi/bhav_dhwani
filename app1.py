import streamlit as st
import librosa
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained model and label encoder
with open('urdu_emotion_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the saved LabelEncoder classes (these should correspond to the emotion labels)
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('classes.npy', allow_pickle=True)

# Mapping the class indices to corresponding emotions manually
emotion_map = {0: 'Happy', 1: 'Sad', 2: 'Neutral', 3: 'Angry'}

# Function to extract MFCC features from audio file
def extract_features(file_path):
    # Load audio file
    audio, sr = librosa.load(file_path, sr=None)
    
    # Extract MFCCs (13 coefficients)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    # Take mean of MFCCs across time axis to represent the whole audio
    mfcc = np.mean(mfcc, axis=1)
    
    return mfcc

# Streamlit app layout
st.title('भाव ध्वनि')

# Upload audio file
uploaded_file = st.file_uploader("Upload an Audio File", type=["wav"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract features from the uploaded audio file
    features = extract_features("temp_audio.wav")
    
    # Reshape the features for model prediction
    features = features.reshape(1, 1, features.shape[0])  # Reshape for LSTM (samples, time steps, features)
    
    # Predict emotion using the trained model
    prediction = model.predict(features)
    
    # Decode the prediction (get the label)
    predicted_index = np.argmax(prediction)
    predicted_label = emotion_map.get(predicted_index, "Unknown")  # Map index to emotion label
    
    # Display the predicted emotion
    st.write(f"Predicted Emotion: {predicted_label}")
