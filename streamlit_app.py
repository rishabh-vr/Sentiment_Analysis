import streamlit as st
from transformers import pipeline

st.title("Fine-Tuning BERT for Twitter Sentiment Classification")

classifier = pipeline('text-classification', model='bert-base-uncased-sentiment-model')

text = st.text_area("Enter Your Tweet Here")

if st.button("Predict"):
        result = classifier(text)
        st.write("Prediction Result:", result)
