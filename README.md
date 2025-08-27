# 📧 SMS / Email Spam Classifier  

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-brightgreen)](https://share.streamlit.io/)  
[![Scikit-Learn](https://img.shields.io/badge/ML-ScikitLearn-orange)](https://scikit-learn.org/stable/)  
[![NLTK](https://img.shields.io/badge/NLP-NLTK-yellowgreen)](https://www.nltk.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  

A machine learning–based web application built with **Streamlit** that classifies SMS or Email messages as **Spam** or **Ham (Not Spam)**.  
This project demonstrates text preprocessing, feature extraction using **TF-IDF**, and classification with **Multinomial Naive Bayes**. 


---

## 🌐 Live Demo
👉 [Click here to try the app](https://sms-spam-classifier-73uytas2eviu6rvgai9dtr.streamlit.app/)  

---

## 🔥 Features
- ✅ Classifies SMS/Email into **Spam** or **Ham** in real time  
- 🎨 Clean and simple **Streamlit web interface**  
- 🔤 Text preprocessing with **NLTK** (stopwords removal, stemming, tokenization)  
- 📊 Feature extraction using **TF-IDF Vectorizer**  
- 🤖 Machine learning model trained with **Multinomial Naive Bayes**  
- 🚀 Deployment-ready with **Streamlit Community Cloud**  

---

## ⚙️ Project Workflow

1. **Data Preprocessing**
   - Lowercasing text  
   - Removing special characters, punctuation, and stopwords  
   - Stemming words  

2. **Feature Engineering**
   - Transform text into numerical vectors using **TF-IDF**  

3. **Model Training**
   - Trained a **MultinomialNB** classifier  
   - Evaluated with accuracy, precision, and recall  

4. **Model Persistence**
   - Saved trained model and vectorizer as `.pkl` files using **pickle**  

5. **Deployment**
   - Interactive UI built with **Streamlit**  
   - Deployed on **Streamlit Cloud**  

---

## 📂 Project Structure
sms-spam-classifier/
│
├── app.py # Streamlit application
├── model.pkl # Trained MultinomialNB model
├── vectorizer.pkl # TF-IDF vectorizer
├── requirements.txt # Dependencies
├── README.md # Project documentation
└── data/ # (Optional) Dataset files



---

## 🚀 Installation & Setup

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/sms-spam-classifier.git
cd sms-spam-classifier

2. Create a virtual environment (recommended)
```bash
Copy code
python -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows
3. Install dependencies
```bash
Copy code   
pip install -r requirements.txt
4. Run Streamlit app
```bash
streamlit run app.py
📊 Example
Input:

pgsql
Copy code
Congratulations! You have won a $1000 Walmart gift card. Click here to claim.
Output:

nginx
Copy code
SPAM 🚨
Input:

rust
Copy code
Hey, are we still meeting for lunch today?
Output:

nginx
Copy code
HAM ✅
🛠️ Tech Stack
Python 3.8+

Streamlit – UI & Deployment

Scikit-learn – Machine Learning Model

NLTK – Text Preprocessing

Pickle – Model persistence

📌 Requirements
Make sure your requirements.txt includes:

nginx
Copy code
streamlit
scikit-learn
nltk
pandas
numpy
🌍 Deployment
Deployed on Streamlit Community Cloud

Push code & model files (app.py, model.pkl, vectorizer.pkl, requirements.txt) to GitHub

Go to Streamlit Cloud → New App

Select repo & branch → Deploy

📜 License
This project is licensed under the MIT License.

