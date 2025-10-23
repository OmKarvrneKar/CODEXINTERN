
import re
import string
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def download_nltk_resources():
    """Download required NLTK resources if not already present."""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print('Downloading NLTK stopwords...')
        nltk.download('stopwords')


def load_dataset(url: str) -> pd.DataFrame:
    """Load the SMS dataset from the provided URL into a DataFrame.

    The file is a tab-separated file with columns: label and message.
    """
    df = pd.read_csv(url, sep='\t', header=0, names=['label', 'message'])
    return df


def preprocess_text(text: str, stemmer: PorterStemmer, stop_words: set) -> str:
    """Preprocess a single message string and return the cleaned text.

    Steps:
    - Lowercase
    - Remove non-alphabetic characters
    - Tokenize by whitespace
    - Remove stopwords
    - Stem tokens
    - Return joined cleaned string
    """
    if not isinstance(text, str):
        return ''

    # Lowercase
    text = text.lower()

    # Remove punctuation and non-alphabetic characters
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Tokenize on whitespace
    tokens = text.split()

    # Remove stopwords and stem
    cleaned = [stemmer.stem(tok) for tok in tokens if tok not in stop_words]

    return ' '.join(cleaned)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform preprocessing on the DataFrame and add `cleaned_message` column."""
    # Ensure downloads
    download_nltk_resources()

    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    # Apply preprocessing
    df['cleaned_message'] = df['message'].apply(lambda x: preprocess_text(x, stemmer, stop_words))
    return df


def vectorize_text(corpus: pd.Series):
    """Convert text corpus to TF-IDF features and return the vectorizer and matrix."""
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=1)
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X


def train_and_evaluate(X_train, X_test, y_train, y_test, model, model_name: str):
    """Train the model, predict on test set, and print evaluation metrics and plot confusion matrix."""
    print(f"\nTraining model: {model_name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {acc:.4f}")

    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])

    # Plot confusion matrix as heatmap
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    return acc


def main():
    # URL for the dataset (tab-separated values)
    DATA_URL = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'

    print('Loading dataset...')
    df = load_dataset(DATA_URL)

    # Display basic information
    print('\nFirst 5 rows of the dataset:')
    print(df.head())

    print(f'\nDataset shape: {df.shape}')

    print('\nLabel distribution:')
    print(df['label'].value_counts())

    # Preprocess text and create cleaned_message column
    print('\nPreprocessing text data (lowercasing, removing non-alpha, stopwords, stemming)...')
    df = prepare_data(df)

    print('\nSample cleaned messages:')
    print(df[['message', 'cleaned_message']].head())

    # Vectorize cleaned text
    print('\nVectorizing text using TF-IDF...')
    vectorizer, X = vectorize_text(df['cleaned_message'])

    # Labels
    y = df['label']

    # Split into train and test sets
    print('\nSplitting data into train and test sets (80% train, 20% test)...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train Multinomial Naive Bayes
    mnb = MultinomialNB()
    mnb_acc = train_and_evaluate(X_train, X_test, y_train, y_test, mnb, 'Multinomial Naive Bayes')

    # Train Logistic Regression
    # Use solver liblinear for small datasets; set max_iter higher to ensure convergence
    logreg = LogisticRegression(solver='liblinear', max_iter=1000)
    logreg_acc = train_and_evaluate(X_train, X_test, y_train, y_test, logreg, 'Logistic Regression')

    # Compare models
    print('\nModel comparison based on accuracy:')
    print(f'MultinomialNB Accuracy: {mnb_acc:.4f}')
    print(f'LogisticRegression Accuracy: {logreg_acc:.4f}')

    if logreg_acc > mnb_acc:
        print('\nLogistic Regression performs better based on accuracy.')
    elif logreg_acc < mnb_acc:
        print('\nMultinomial Naive Bayes performs better based on accuracy.')
    else:
        print('\nBoth models perform equally based on accuracy.')


if __name__ == '__main__':
    main()
