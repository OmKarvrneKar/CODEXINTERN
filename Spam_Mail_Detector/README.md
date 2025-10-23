# Spam Mail Detector

This project trains and evaluates text classifiers (Multinomial Naive Bayes and Logistic Regression) to detect spam messages using the SMS Spam Collection Dataset.

Files:
- `spam_mail_detector.py` - Main script. Downloads dataset at runtime, preprocesses text, trains models, and shows evaluation metrics and confusion matrices.
- `requirements.txt` - Python package requirements.

Quick start (Windows PowerShell):

1. Install Python 3.8+ and ensure `python` is on your PATH. On Windows you can install from https://www.python.org/downloads/.

2. From this project folder, create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

3. Run the script:

```powershell
python .\spam_mail_detector.py
```

Notes:
- The script will download NLTK stopwords at first run if they're missing.
- If `python` is not available on PATH, you can either add it to PATH or run the installer and select the "Add Python to PATH" option.
