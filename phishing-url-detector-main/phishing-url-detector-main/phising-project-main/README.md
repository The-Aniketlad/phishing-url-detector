# Phishing URL Detection System

This project detects whether a given URL is **Phishing** or **Legitimate**
using Machine Learning (Random Forest + Neural Network).

## Features
- URL-based phishing detection
- Dual-model prediction (NN + RF)
- Safe alternative link suggestions
- Modern UI (Tailwind CSS)
- Flask REST API backend

## Tech Stack
- Python
- Flask
- Scikit-learn
- TensorFlow
- HTML + Tailwind CSS

## How to Run

```bash
# create virtual environment
python -m venv venv

# activate venv
venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# run backend
cd phising-project-main/backend
python app.py
