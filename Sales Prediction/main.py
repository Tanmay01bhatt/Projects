import streamlit as st
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open('trained_model.pkl', 'rb'))

def prediction(TV,Radio,Newspaper):
    pred = model.predict([[TV,Radio,Newspaper]])
    return pred

def main():
    st.title("Sales Prediction")
    tv = st.text_input("TV")
    radio = st.text_input("Radio")
    news = st.text_input("Newspaper")
    if st.button("Predict"):
        out = prediction(tv,radio,news)
        st.text(out)

if __name__ == '__main__' :
    main()