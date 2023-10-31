import streamlit as st
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open('classifier.pkl', 'rb'))

def prediction(variance,skewness,curtosis,entropy):
    pred = model.predict([[variance,skewness,curtosis,entropy]])
    return pred

def main():
    st.title("Bank Note Authenticaion")
    variance = st.text_input("Variance")
    skewness = st.text_input("Skewness")
    curtosis = st.text_input("curtosis")
    entropy = st.text_input("entropy")
    if st.button("Predict"):
        out = prediction(variance,skewness,curtosis,entropy)
        st.text(out)

if __name__ == '__main__' :
    main()