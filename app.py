from flask import Flask, request, jsonify, render_template
import numpy as np
import librosa
import tensorflow as tf
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Bird class mapping
BIRD_CLASSES = {
    0: "Sparrow",
    1: "Robin",
    2: "Blue Jay",
    3: "Cardinal",
    4: "Goldfinch",
    5: "Woodpecker"
}

# Function to extract MFCC features
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)  # Load audio
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # Extract MFCC features
    mfccs = np.mean(mfccs, axis=1)  # Average over time
    return mfccs.reshape(1, -1)  # Reshape for model

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    file_path = "temp_audio.wav"
    file.save(file_path)  # Save the file temporarily

    try:
        features = extract_features(file_path)  # Extract MFCC features
        prediction = model.predict(features)
        predicted_index = np.argmax(prediction)  # Get predicted class index
        bird_name = BIRD_CLASSES.get(predicted_index, "Unknown Bird")  # Map to bird name
    except Exception as e:
        try:
           os.remove(file_path)
        except FileNotFoundError:
                pass  # Ignore if already deleted


    os.remove(file_path)  # Clean up temp file

    return jsonify({"bird": bird_name})

if __name__ == "__main__":
    app.run(debug=True)
