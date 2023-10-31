import streamlit as st
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open('trained_model.pkl', 'rb'))
tf_idf = pickle.load(open('tf_idf.pkl', 'rb'))

def prediction(a):
  a = np.array([a])
  #Vectorization using TFIDF
  a = tf_idf.transform(a)
  #PREDICTED VALUE
  x = model.predict(a)
  return x

st.title("Cuisine Detection")
inp = st.text_input('Text')

if st.button('Detect'):
    out = prediction(inp)
    st.text(out)
