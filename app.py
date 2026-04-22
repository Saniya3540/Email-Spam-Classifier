import streamlit as st
import pickle
import re
import nltk
from nltk.stem.porter import PorterStemmer

# Fix for stopwords
try:
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))

# load model
model = pickle.load(open("model.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))

ps = PorterStemmer()

def predict_spam(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower().split()
    
    review = [ps.stem(word) for word in review if word not in stop_words]
    review = ' '.join(review)
    
    vector = cv.transform([review]).toarray()
    result = model.predict(vector)[0]
    
    return "🚨 Spam" if result == 1 else "✅ Ham"

st.title("📩 Email Spam Detection System")

msg = st.text_area("Enter message")

if st.button("Predict"):
    result = predict_spam(msg)
    st.success(result)
