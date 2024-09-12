# Next Word Prediction with LSTM

This project is a web-based application that predicts the next word in a sequence using an LSTM (Long Short-Term Memory) model. The application is built using Streamlit, a Python framework for creating interactive web applications.

## Features
- **Interactive UI**: Users can input a sequence of words, and the app predicts the next word in the sequence.
- **LSTM Model**: The model used for prediction is a pre-trained LSTM neural network.
- **Tokenizer Integration**: The app uses a tokenizer to convert text to sequences, which are fed into the model for prediction.

## Installation

To run this project locally, follow the steps below:

### Clone the repository
```bash
git clone https://github.com/DIXIT0043/Predict_Next_Word_LSTM.git
```
## Install dependencies

Ensure you have Python 3.x installed. You can install the required packages using pip:

```bash

pip install streamlit tensorflow numpy pandas
```
## Download the Model and Tokenizer

Make sure to place the following files in the correct paths:

    LSTM Model: next_word_prediction_model.h5
    Tokenizer: tokenizer_next_word_prediction.pkl

You need to modify the file paths in app.py if your model and tokenizer are located in different directories.
## Run the Application

To run the Streamlit app, use the following command:

```bash

streamlit run app.py
```
## How It Works

Model Loading: The LSTM model (next_word_prediction_model.h5) is loaded using Keras.
Tokenizer Loading: The tokenizer (tokenizer_next_word_prediction.pkl) is loaded using pickle.
Prediction: The app takes a user input of a sequence of words and uses the model to predict the next word.

## File Structure

app.py: The main script that runs the Streamlit web app.
next_word_prediction_model.h5: The pre-trained LSTM model.
tokenizer_next_word_prediction.pkl: The tokenizer used to preprocess the input text.

## Example Usage

Input a sequence of words in the provided text box.
Click on the "Predict Next Button".
The predicted next word will be displayed on the screen.

# Requirements

Python 3.x
TensorFlow
Streamlit
NumPy
Pandas
