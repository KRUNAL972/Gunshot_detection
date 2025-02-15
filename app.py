import os
import tempfile
import numpy as np
import librosa
import tensorflow as tf
import streamlit as st

# Define your GunshotDetector class
class GunshotDetector:
    def __init__(self, model_path="E:\\gunshot\\gunshot_detection_model.h5", n_mfcc=13, max_pad_length=40):
        """
        Initializes the gunshot detector with the given model and MFCC parameters.
        Args:
            model_path (str): Path to the saved Keras model.
            n_mfcc (int): Number of MFCC features to extract.
            max_pad_length (int): Maximum length for padding/truncating the MFCC feature array.
        """
        self.model_path = model_path
        self.n_mfcc = n_mfcc
        self.max_pad_length = max_pad_length
        self.model = self.load_model()

    def load_model(self):
        """Loads the Keras model from disk."""
        if not os.path.exists(self.model_path):
            raise ValueError(f"Model not found at: {self.model_path}")
        model = tf.keras.models.load_model(self.model_path)
        return model

    def extract_features(self, audio_path):
        """
        Extracts MFCC features from an audio file.
        Args:
            audio_path (str): Path to the audio file.
        Returns:
            np.ndarray: A flattened array of MFCC features padded or truncated to a fixed length.
        """
        try:
            audio, sample_rate = librosa.load(audio_path, res_type="kaiser_fast")
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=self.n_mfcc)
            # Pad or truncate the MFCC array to have a fixed number of time steps.
            pad_width = self.max_pad_length - mfccs.shape[1]
            if pad_width > 0:
                mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode="constant")
            else:
                mfccs = mfccs[:, :self.max_pad_length]
            return mfccs.flatten()
        except Exception as e:
            st.error(f"Error processing {audio_path}: {e}")
            return None

    def predict(self, audio_path):
        """
        Predicts whether the audio file contains a gunshot.
        Args:
            audio_path (str): Path to the audio file.
        Returns:
            str: "Gunshot detected" if a gunshot is predicted, otherwise "No gunshot detected".
        """
        features = self.extract_features(audio_path)
        if features is None:
            return "Error processing audio file."
        # Reshape features for model input (1 sample, many features)
        features = features.reshape(1, -1)
        prediction = self.model.predict(features)
        # Using a threshold of 0.5 for binary classification
        if prediction[0][0] > 0.5:
            return "Gunshot detected"
        else:
            return "No gunshot detected"

# ---------------------------
# Streamlit UI begins here
# ---------------------------
st.set_page_config(page_title="Gunshot Detection System", layout="wide")
st.title("Gunshot Detection System")
st.markdown("""
This web application detects gunshots in an uploaded audio file using a pre-trained deep learning model.
Upload an audio file (WAV, MP3, OGG, FLAC) and the system will analyze it to determine if a gunshot sound is present.
""")

# Sidebar with additional information
st.sidebar.title("About")
st.sidebar.info("""
This app uses a deep learning model to detect gunshots in audio. 
The model was trained on the UrbanSound8K dataset using MFCC features and a feedforward neural network.
When a gunshot is detected, an alert is sent to +919356684307.
""")

# File uploader widget; adjust accepted file types as needed.
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg", "flac"])

if uploaded_file is not None:
    # Write the uploaded file to a temporary file.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    st.write("### Processing the audio file...")
    
    try:
        # Initialize the detector (update model_path if needed)
        detector = GunshotDetector(model_path="E:\\gunshot\\gunshot_detection_model.h5")
        # Run prediction on the temporary file
        result = detector.predict(temp_file_path)
        
        # Display result with additional alert message if gunshot is detected
        if result == "Gunshot detected":
            st.success("Gunshot detected! Sending alert to +919356684307")
        elif result == "No gunshot detected":
            st.info("No gunshot detected in the audio.")
        else:
            st.error(result)
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        # Optionally remove the temporary file
        os.remove(temp_file_path)

# Additional UI enhancements (footer, contact info, etc.)
st.markdown("---")
st.markdown("""
**Note:** This is a demonstration application. In a production system, the alert mechanism would integrate with an SMS gateway to send real alerts.
""")
