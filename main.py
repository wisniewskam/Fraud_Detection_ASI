import os
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from autogluon.tabular import TabularPredictor
from gcs_utils import download_and_extract_model

MODEL_DIR = "model_predictor"
BUCKET_NAME = "my-fraud-detector"
MODEL_NAME = "predictor.zip"

if not os.path.exists(MODEL_DIR):
    download_and_extract_model(BUCKET_NAME, MODEL_NAME, MODEL_DIR)

model = TabularPredictor.load(MODEL_DIR)

app = FastAPI()

class InputData(BaseModel):
    features: list

@app.get("/")
def root():
    return {"status": "API działa"}

@app.post("/predict")
def predict(data: InputData):
    x = np.array([data.features])
    prediction = model.predict(x)
    return {"prediction": prediction.tolist()}

print(model.features())

