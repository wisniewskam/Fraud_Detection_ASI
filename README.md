# Fraud Detection – Bank Transaction Classifier

This project is a modular fraud detection system built using Kedro, trained with AutoGluon (AutoML), and deployed with a FastAPI + Gradio interface. The application can be used to predict whether a given bank transaction is fraudulent.

## Features
- Modular pipeline architecture (Kedro)
- AutoML-based classifier (AutoGluon)
- Cloud model storage (Google Cloud Storage)
- Containerized via Docker
- REST API with FastAPI (Uvicorn)
- Visual interface with Gradio (for users)


## How to Run Locally

### 1. Install dependencies
Make sure you have Python 3.10 installed (AutoGluon is version-sensitive).

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

```bash
python main.py
```

## Access the App
REST API: http://localhost:8000
Gradio Web UI: http://localhost:8000/gradio

## Docker support
To build and run the app in a container (make sure Docker is active):

```bash
docker build -t fraud-detector-app .
docker run -p 8000:8000 fraud-detector-app
```
