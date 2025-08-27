# import streamlit as st
# import pickle
# import string
# from nltk.corpus import stopwords
# import nltk
# from nltk.stem.porter import PorterStemmer
# import pickle
#
# # Open the file in read-binary ('rb') mode
# with open('vectorizer.pkl', 'rb') as f:
#     vectorizer = pickle.load(f)
#
# # Now you can use the 'vectorizer' object in your code
# # For example:
# # new_data_transformed = vectorizer.transform(["some new sms text"])
#
# print(vectorizer)
#
#
# ps = PorterStemmer()
#
#
# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)
#
#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)
#
#     text = y[:]
#     y.clear()
#
#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)
#
#     text = y[:]
#     y.clear()
#
#     for i in text:
#         y.append(ps.stem(i))
#
#     return " ".join(y)
#
# tfidf = pickle.load(open('vectorizer.pkl','rb'))
# model = pickle.load(open('model.pkl','rb'))
#
# st.title("Email/SMS Spam Classifier")
#
# input_sms = st.text_area("Enter the message")
#
# if st.button('Predict'):
#
#     # 1. preprocess
#     transformed_sms = transform_text(input_sms)
#     # 2. vectorize
#     vector_input = tfidf.transform([input_sms]).toarray()
#     # 3. predict
#     result = model.predict(vector_input)[0]
#     # 4. Display
#     if result == 1:
#         st.header("Spam")
#     else:
#         st.header("Not Spam")
#

import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

import nltk

# Ensure required NLTK data is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


# Load saved vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

ps = PorterStemmer()

# Text preprocessing
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# ------------------- UI Styling -------------------
st.set_page_config(page_title="Spam Classifier", page_icon="✉️", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;70_0&display=swap');
        body {
            background-color: #f0f2f6;
        }
        .stApp {
            background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100' viewBox='0 0 100 100'%3E%3Cg fill-rule='evenodd'%3E%3Cg fill='%239C92AC' fill-opacity='0.1'%3E%3Cpath opacity='.5' d='M96 95h4v1h-4v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4H0v-1h4v-9H0v-1h4v-9H0v-1h4v-9H0v-1h4v-9H0v-1h4v-9H0v-1h4v-9H0v-1h4v-9H0v-1h4v-9H0v-1h4V0h1v4h9V0h1v4h9V0h1v4h9V0h1v4h9V0h1v4h9V0h1v4h9V0h1v4h9V0h1v4h9V0h1v4h4v1h-4v9h4v1h-4v9h4v1h-4v9h4v1h-4v9h4v1h-4v9h4v1h-4v9h4v1h-4v9h4v1h-4v9zm-1 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-9-10h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm9-10v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-9-10h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm9-10v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-9-10h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm9-10v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-9-10h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9z'/%3E%3Cpath d='M6 5V0h1v5h9V0h1v5h9V0h1v5h9V0h1v5h9V0h1v5h9V0h1v5h9V0h1v5h9V0h1v5h9V0h1v5h4v1h-4v9h4v1h-4v9h4v1h-4v9h4v1h-4v9h4v1h-4v9h4v1h-4v9h4v1h-4v9h4v1h-4v9h4v1h-4v9h-1v-9h-9v9h-1v-9h-9v9h-1v-9h-9v9h-1v-9h-9v9h-1v-9h-9v9h-1v-9h-9v9h-1v-9h-9v9h-1v-9h-9v9H5v-1h1v-9H5v-1h1v-9H5v-1h1v-9H5v-1h1v-9H5v-1h1v-9H5v-1h1v-9H5v-1h1v-9H5v-1h1V5h-1z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        }
        .title {
           font-size: 2.5rem;
            font-weight: 700;
            color: #1a2a6c;
            text-align: center;
            margin-bottom: 0.5rem;
        }
      .subheader {
            text-align: center;
            font-size: 1.1rem;
            color: #555;
            margin-bottom: 2.5rem;
        }
        .stTextArea textarea {
            border-radius: 10px;
            border: 2px solid #b21f1f;
            background-color: #ffffff;
            font-family: 'Poppins', sans-serif;
            font-size: 1rem;
            transition: all 0.3s ease;
        }

        .stTextArea textarea:focus {
            border-color: #1a2a6c;
            box-shadow: 0 0 0 3px rgba(26, 42, 108, 0.2);
        }
         .stButton>button {
            background: linear-gradient(to right, #b21f1f, #1a2a6c);
            color: white;
            font-weight: 600;
            font-family: 'Poppins', sans-serif;
            border-radius: 10px;
            padding: 0.75em 2em;
            border: none;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px 0 rgba(26, 42, 108, 0.3);
            cursor: pointer;
            width: 100%; /* Make button full width */
        }

        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px 0 rgba(26, 42, 108, 0.4);
        }
         .result {
            margin-top: 2rem;
            padding: 1rem;
            border-radius: 12px;
            text-align: center;
            font-size: 1.5rem;
            font-weight: 600;
            font-family: 'Poppins', sans-serif;
            transition: all 0.3s ease;
        }

        .spam {
            background-color: #ffdde1;
            color: #d32f2f;
            border: 1px solid #d32f2f;
        }

        .not-spam {
            background-color: #e8f5e9;
            color: #388e3c;
            border: 1px solid #388e3c;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            padding: 1rem;
        }

    </style>
""", unsafe_allow_html=True)

# ------------------- MAIN APP -------------------
st.markdown("<div class='title'>Email/SMS Spam Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Detect whether a message is Spam or Not Spam instantly!</div>", unsafe_allow_html=True)

input_sms = st.text_area("✍️ Enter your message below:")

if st.button(' Predict '):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([input_sms]).toarray()
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.markdown("<div class='result spam'>🚨 Spam</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result not-spam'>✅ Not Spam</div>", unsafe_allow_html=True)

# ------------------- SIDEBAR -------------------
st.sidebar.title("ℹ️ About This App")
st.sidebar.write(
    """
    This **Spam Classifier App** uses **Machine Learning (NLP + SVM/Ensemble)**  
    to detect whether an Email or SMS is **Spam** or **Not Spam**.  

    ### 🔧 How it Works
    1. Preprocessing: Cleans the text (lowercase, remove stopwords, stemming).  
    2. Vectorization: Converts text into numbers using **TF-IDF**.  
    3. Prediction: Classifies using the trained ML model.  

    ### 📌 Features
    - Fast & Accurate Detection  
    - Works on both **Emails & SMS**  
    - Simple & Beautiful UI  

    ---
    👨‍💻 Developed by: **ANAND KUMAR**
    """
)



