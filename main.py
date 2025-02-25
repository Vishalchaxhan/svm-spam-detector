import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
data = pd.read_csv("spam.csv", encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Feature extraction
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)

y_pred = svm_model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit UI
st.title("SMS Spam Detector")
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

user_input = st.text_area("Enter a message:")
if st.button("Check Spam"):
    input_tfidf = vectorizer.transform([user_input])
    prediction = svm_model.predict(input_tfidf)[0]
    result = "Spam" if prediction == 1 else "Not Spam"
    st.write(f"Result: {result}")



def model_details():
    report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])
    matrix = confusion_matrix(y_test, y_pred)

    st.subheader("Model Performance Details")
    st.text("Classification Report:")
    st.text(report)

    st.subheader("Confusion Matrix")
    st.write(matrix)

if st.button("Show Model Details"):
    model_details()


