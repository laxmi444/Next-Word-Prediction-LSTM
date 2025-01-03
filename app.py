import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# load the model and tokenizer
model= load_model("C:\\Users\\Omkar\\next_word_pred\\next_word_pred_lstm.h5")

with open("C:\\Users\\Omkar\\next_word_pred\\tokenizer", "rb") as handle:
    tokenizer = pickle.load(handle)


# function
def predict_next_word(model, tokenizer, text, max_sequence_length):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_length:
        token_list = token_list[-(max_sequence_length - 1):]  # ensure sequence length matches max_sequence
    token_list = pad_sequences([token_list], maxlen=max_sequence_length - 1, padding="pre")
    
    # predict the next word probabilities
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]  # Extract the predicted index
    
    # find the corresponding word
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None  # rEturn None if no match is found

# streamlit app
st.title("Next Word Prediction with LSTM")
input_text=st.text_input("Enter the sequence of words","Your text")
st.button("Predict next word")
if st.button("Predict next word"):
    max_sequence_length=model.input_shape[1]+1
    next_word= predict_next_word(model, tokenizer, input_text, max_sequence_length)
    st.write(f"Next word prediction:{next_word}")
