import os
import numpy as np
import librosa
import tensorflow as tf

class GunshotDetector:
    def __init__(self, model_path="gunshot_detection_model.h5", n_mfcc=13, max_pad_length=40):
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
            print(f"Error processing {audio_path}: {e}")
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

# Optional: Add a command-line interface to use the module easily.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gunshot detection using a pre-trained model.")
    parser.add_argument("--audio_path", required=True, help="E:\gunshot\deagle.mp3")
    parser.add_argument("--model_path", default="E:\gunshot\gunshot_detection_model.h5", help="Path to the saved model file.")
    args = parser.parse_args()

    detector = GunshotDetector(model_path=args.model_path)
    result = detector.predict(args.audio_path)
    print(result)


