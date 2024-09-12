import pandas as pd
import numpy as np
import pickle
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model  = load_model('next_word_prediction_model.h5')

with open('tokenizer_next_word_prediction.pkl','rb') as handle:
    tokeizer = pickle.load(handle)

def predict_next_word(model,tokenizer,text,max_seq_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list)>=max_seq_len:
        token_list = token_list[-(max_seq_len-1):]
    token_list=pad_sequences([token_list],maxlen=max_seq_len-1,padding='pre')
    predicted = model.predict(token_list,verbose=0)
    predicted_word_index = np.argmax(predicted,axis=1)
    for word,index in tokenizer.word_index.items():
        if index== predicted_word_index:
            return word
    return None


st.title('Next Word Prediction with LSTM')
input_text = st.text_input('Enter the Sequnce of Words')
if st.button('Predict Next Button'):
    max_sequence_len = model.input_shape[1]+1
    next_word = predict_next_word(model=model,tokenizer=tokeizer,text=input_text,max_seq_len=max_sequence_len)
    st.write(f'Next Word is : {next_word}')


