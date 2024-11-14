import gradio as gr
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize  
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import nltk

# Download NLTK resources (run once)
nltk.download('stopwords')
nltk.download('punkt')

# Function for text preprocessing
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove non-alphanumeric characters
    text = re.sub(r'\W', ' ', text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('indonesian'))
    tokens = [word for word in tokens if word not in stop_words]
    # Apply stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    # Join tokens back into text
    preprocessed_text = ' '.join(stemmed_tokens)
    return preprocessed_text

# Load the pre-trained Logistic Regression model and TfidfVectorizer
with open('logistic_model', 'rb') as f:
    logistic_regression_model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Function for prediction
def classify_news(text_input):
    # Preprocess the text
    preprocessed_text = preprocess_text(text_input)
    # Vectorize the preprocessed text
    X_text = vectorizer.transform([preprocessed_text])
    # Make prediction
    prediction = logistic_regression_model.predict(X_text)
    # Return the prediction result
    return "Prediksi Berita: Travel News" if prediction[0] == 1 else "Prediksi Berita: Health News"

# Set up Gradio interface
interface = gr.Interface(fn=classify_news, inputs="text", outputs="text",
                         title="Text Preprocessing and Classification App",
                         description="Aplikasi ini menggunakan preprocessing text dan membuat prediksi menggunakan pre-trained Logistic Regression Model.")

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch()
