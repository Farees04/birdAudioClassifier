# Bird Sound Classification Project

This project aims to classify bird species based on their sound. The model is built using deep learning techniques, specifically using a Convolutional Neural Network (CNN), and the trained model is saved as `model.h5`. The project uses a custom dataset for bird sound classification, and the dataset is sourced from Kaggle.

## Project Flow

1. **Data Preprocessing and Model Training**: 
   - The project starts with data preprocessing and model training in the `classifier.ipynb` Jupyter Notebook.
   - The notebook trains a CNN model to classify 5 bird species based on their sound data.
   - After training, the model is saved as `model.h5` for further use.

2. **Running the Web App**:
   - You can run the web application using the `app.py` file. This web app allows you to upload your own bird sound files and get predictions based on the trained model.
   - The app provides a user-friendly interface where you can upload an audio file, and the model will predict the bird species for you.

3. **Dataset**: 
   - The dataset used for training is a collection of bird sound recordings from Kaggle.
   - It contains audio files labeled with the bird species, which are used to train the model to recognize specific bird sounds.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Farees04/birdAudioClassifier.git
   cd bird-sound-classifier
   ```
   
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
    ```
   
4. Train the model:
    Open `classifier.ipynb` in Jupyter Notebook and run all the cells to train the model.
    After training, the model will be saved as `model.h5`.
   
5. Run the Flask web app:
   ```bash
   python app.py
   ```
6. Navigate to `http://127.0.0.1:5000` in your browser to access the app and classify your own bird sound dataset

## Usage
**Input**: Upload an audio file.

**Output**: The model will predict the bird species based on the uploaded audio file and display the result on the webpage.
