from flask import Flask, request, jsonify, render_template
import numpy as np
import librosa
import tensorflow as tf
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("model.h5")


bird_list = ["Bewick's Wren", "Northern Mockingbird",
              "American Robin", "Song Sparrow", "Northern Cardinal"]




def features_extractor(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
    
    return mfccs_scaled_features


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
        features = features_extractor(file_path)  # Extract MFCC features
        normalized_data = tf.keras.utils.normalize(features)
        prediction = model.predict(normalized_data)

        predicted_class = np.argmax(prediction, axis=1)
        bird_name = bird_list[predicted_class[0]]
         # Get predicted class index
    
    except Exception as e:
        try:
           os.remove(file_path)
        except FileNotFoundError:
                pass  # Ignore if already deleted


    os.remove(file_path)  # Clean up temp file

    return {"bird":bird_name}

if __name__ == "__main__":
    app.run(debug=True)
