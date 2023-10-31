import streamlit as st
import numpy as np
import pandas as pd
import pickle
import random

model = pickle.load(open('history.p', 'rb'))
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
unique_tokens = pickle.load(open('unique_tokens', 'rb'))
unique_token_index = pickle.load(open('unique_token_index', 'rb'))
n_words =15
def predict_next_word(input_text, n_best):
    input_text = input_text.lower()
    X = np.zeros((1, n_words, len(unique_tokens)))# 1 input sentence
    for i, word in enumerate(input_text.split()):
        X[0, i, unique_token_index[word]] = 1   #check if each word is present in unique words index

    predictions = model.predict(X)[0]
    return np.argpartition(predictions, -n_best)[-n_best:]

def generate_text(input_text, n_words, most_possible_choice=3):
    word_sequence = input_text.split()
    current = 0
    for _ in range(n_words):
        sub_sequence = " ".join(tokenizer.tokenize(" ".join(word_sequence).lower())[current:current+n_words])
        try:
            choice = unique_tokens[random.choice(predict_next_word(sub_sequence,most_possible_choice))]
        except:
            choice = random.choice(unique_tokens)
        word_sequence.append(choice)
        current += 1
    return " ".join(word_sequence)

st.title("Prediction")
text = st.text_input("text")

if st.button("Predict"):
    out = generate_text(text,15,5)
    st.text(out)