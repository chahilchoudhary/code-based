import streamlit as st
import pandas as pd
import re
import pickle
from PyPDF2 import PdfReader
import docx2txt

# Load the saved model, vectorizer, and label encoder using pickle
def load_model():
    with open('sentiment_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    with open('label_encoder.pkl', 'rb') as le_file:
        label_encoder = pickle.load(le_file)
    return model, vectorizer, label_encoder

model, vectorizer, label_encoder = load_model()

# Function to clean the text data
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'<.*?>', '', text)    # remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove special characters and punctuation
    text = text.lower()  # convert to lowercase
    return text

# Function to read uploaded files
def read_file(file):
    if file.type == "text/plain":
        text = file.read().decode("utf-8")
    elif file.type == "application/pdf":
        pdf_reader = PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = docx2txt.process(file)
    else:
        text = None
    return text

# Function to predict sentiment
def predict_sentiment(text):
    clean_input = clean_text(text)
    input_vector = vectorizer.transform([clean_input]).toarray()
    prediction = model.predict(input_vector)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    return predicted_label

# Streamlit app interface
def main():
    st.title("Hinglish Sentiment Analysis")

    # Section to upload a file for sentiment prediction
    st.header("Upload a File")
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx"])

    if uploaded_file is not None:
        text = read_file(uploaded_file)
        if text:
            st.write("File content:")
            st.text_area("Text Preview", text, height=200)

            # Predict sentiment for each comment
            st.write("Sentiment Predictions:")
            comments = text.split("\n")  # Assuming each comment is on a new line
            for comment in comments:
                if comment.strip():
                    sentiment = predict_sentiment(comment)
                    st.write(f"Comment: {comment.strip()}")
                    st.write(f"Predicted Sentiment: {sentiment}")
                    st.write("---")
        else:
            st.error("Could not read the file. Please upload a valid file.")

    # Section for text input and sentiment prediction
    st.header("Enter a Comment")
    user_input = st.text_area("Write a comment here", height=100)

    if st.button("Predict Sentiment"):
        if user_input.strip():
            sentiment = predict_sentiment(user_input)
            st.write(f"The predicted sentiment is: **{sentiment}**")
        else:
            st.error("Please write a valid comment.")

if __name__ == '__main__':
    main()
