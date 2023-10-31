import streamlit as st
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open('trained_model.pkl', 'rb'))
cv = pickle.load(open('cv.pkl', 'rb'))


def predict(text):
    x = cv.transform([text]).toarray()  # converting text to bag of words model (Vector)
    lang = model.predict(x)  # predicting the language
    return lang[0]  # printing the language


st.title("Language Identification")
inp = st.text_input('Text')

if st.button('Predict'):
    out = predict(inp)
    st.text(out)