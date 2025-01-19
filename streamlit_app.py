import streamlit as st
from transformers import pipeline

st.title("Fine-Tuning BERT for Movie-Review Sentiment Classification")

classifier = pipeline('text-classification', model='bert-base-uncased-sentiment-model')

text = st.text_area("Enter Your Review Here")

if st.button("Predict"):
        result = classifier(text)
        st.write("Prediction Result:", result)
