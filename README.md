# SMS Spam Classifier

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2%2B-orange.svg)
![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

A machine learning project to accurately classify SMS messages as either "spam" or "ham" (not spam). This project uses Natural Language Processing (NLP) techniques to preprocess the text data and a classification model to make predictions.

## Table of Contents

-   [Project Overview](#project-overview)
-   [Dataset](#dataset)
-   [Workflow](#workflow)
-   [Technologies Used](#technologies-used)
-   [Setup and Installation](#setup-and-installation)
-   [How to Run](#how-to-run)
-   [Results and Evaluation](#results-and-evaluation)
-   [File Structure](#file-structure)
-   [License](#license)

## Project Overview

The goal of this project is to build a reliable model that can distinguish between legitimate SMS messages (ham) and unsolicited, often malicious, messages (spam). The process involves cleaning and transforming raw text data into a numerical format that a machine learning algorithm can understand, and then training a model on this data.

## Dataset

The model is trained on the **SMS Spam Collection Dataset** from the UCI Machine Learning Repository.

-   **Source:** [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
-   **Description:** The dataset contains 5,572 SMS messages in English, tagged according to being `ham` or `spam`.
-   **Columns:**
    -   `label`: The target variable (`ham` or `spam`).
    -   `message`: The raw text of the SMS message.

## Workflow

The project follows a standard machine learning pipeline for NLP:

1.  **Data Loading:** The dataset is loaded into a Pandas DataFrame.
2.  **Data Cleaning and Preprocessing:**
    -   Text is converted to lowercase.
    -   Special characters and punctuation are removed.
    -   The text is tokenized into individual words.
    -   Stop words (common words like 'the', 'is', 'a') are removed.
    -   Stemming is applied to reduce words to their root form (e.g., 'running' becomes 'run').
3.  **Feature Extraction:** The preprocessed text data is converted into numerical vectors using the **TF-IDF (Term Frequency-Inverse Document Frequency)** technique. This creates a matrix where each row represents a message and each column represents a word's importance.
4.  **Model Building:** A classification algorithm is chosen to learn from the feature vectors. For this project, a **Multinomial Naive Bayes** classifier was used due to its effectiveness in text classification tasks.
5.  **Model Training:** The dataset is split into training and testing sets (e.g., 80% for training, 20% for testing), and the model is trained on the training data.
6.  **Model Evaluation:** The model's performance is evaluated on the unseen test set using metrics like Accuracy, Precision, Recall, and F1-Score.

## Technologies Used

-   **Programming Language:** Python 3.9
-   **Libraries:**
    -   **Scikit-learn:** For machine learning models (TF-IDF, Naive Bayes) and evaluation metrics.
    -   **Pandas:** For data manipulation and analysis.
    -   **NLTK (Natural Language Toolkit):** For text preprocessing tasks like tokenization, stop word removal, and stemming.
    -   **Matplotlib / Seaborn:** For data visualization (e.g., confusion matrix).
    -   **Joblib / Pickle:** For saving and loading the trained model and vectorizer.
    -   **Streamlit / Flask (Optional):** If you built a web interface to interact with the model.

## Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK data:**
    Run the following command in a Python shell to download the necessary NLTK packages.
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

## How to Run

To make a prediction on a new SMS message, you can run the main application script:


# Example if you have a Streamlit app
streamlit run app.py

# Example if you have a simple command-line script
python script_runner.py

Results and Evaluation
The model was evaluated on the test set, achieving the following performance:

Accuracy: XX.XX% (e.g., 98.56%)

Precision: XX.XX% (e.g., 99.10%)

Recall: XX.XX% (e.g., 94.25%)

A confusion matrix was also generated to visualize the model's performance in distinguishing between the two classes.

File Structure
```bash
    spam-sms-classifier/
    │
    ├── app.py                  # Main script for the web app (e.g., Streamlit/Flask)
    ├── model.pkl               # Saved trained classification model
    ├── vectorizer.pkl          # Saved TF-IDF vectorizer
    ├── script_runner.py        # Example script to run predictions from the command line
    ├── requirements.txt        # List of Python dependencies
    ├── .gitignore              # Files and directories to be ignored by Git
    └── README.md               # Project documentation
```
License
This project is licensed under the MIT License. See the LICENSE file for more details.








