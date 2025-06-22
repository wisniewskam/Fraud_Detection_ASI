import os
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from autogluon.tabular import TabularPredictor

MODEL_DIR = "model_predictor"
BUCKET_NAME = "my-fraud-detector"
MODEL_NAME = "model_predictor.zip"

from gcs_utils import download_and_extract_model
if not os.path.exists(MODEL_DIR):
    download_and_extract_model(BUCKET_NAME, MODEL_NAME, MODEL_DIR)

predictor = TabularPredictor.load(MODEL_DIR, require_py_version_match=False)

FEATURE_COLUMNS = [
    "Gender", "Age", "State", "City", "Account_Type", "Transaction_Date", "Transaction_Time",
    "Transaction_Amount", "Transaction_Type", "Merchant_Category", "Account_Balance",
    "Transaction_Device", "Transaction_Location", "Device_Type", "Transaction_Description"
]

app = FastAPI(title="Bank Transaction Fraud Detector")

class InputRow(BaseModel):
    data: Dict[str, Any]

@app.get("/")
def root():
    return {
        "status": "running",
        "ui": "/gradio",
        "endpoints": ["/features", "/predict"],
    }

@app.get("/features", response_model=list[str])
def features():
    return FEATURE_COLUMNS

@app.post("/predict")
def predict(item: InputRow):
    df = pd.DataFrame([item.data])
    label = int(predictor.predict(df).iloc[0])
    proba = predictor.predict_proba(df).iloc[0].to_dict()
    return {"prediction": label, "probabilities": proba}

TYPE_CHOICES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
ACCOUNT_TYPE_CHOICES = ["Savings", "Current", "Salary", "Business"]
MERCHANT_CATEGORY_CHOICES = ["Grocery", "Electronics", "Clothing", "Restaurants", "Others"]
DEVICE_TYPE_CHOICES = ["Mobile", "Desktop", "Tablet"]

def infer(
    Gender, Age, State, City, Account_Type, Transaction_Date, Transaction_Time,
    Transaction_Amount, Transaction_Type, Merchant_Category, Account_Balance,
    Transaction_Device, Transaction_Location, Device_Type, Transaction_Description
):
    row = {
        "Gender": Gender,
        "Age": Age,
        "State": State,
        "City": City,
        "Account_Type": Account_Type,
        "Transaction_Date": Transaction_Date,
        "Transaction_Time": Transaction_Time,
        "Transaction_Amount": Transaction_Amount,
        "Transaction_Type": Transaction_Type,
        "Merchant_Category": Merchant_Category,
        "Account_Balance": Account_Balance,
        "Transaction_Device": Transaction_Device,
        "Transaction_Location": Transaction_Location,
        "Device_Type": Device_Type,
        "Transaction_Description": Transaction_Description,
    }
    df = pd.DataFrame([row])
    label = int(predictor.predict(df).iloc[0])
    proba = predictor.predict_proba(df).iloc[0].to_dict()
    verdict = "❌ Oszustwo" if label == 1 else "✅ Transakcja bezpieczna"
    return verdict, proba

with gr.Blocks(title="Weryfikacja transakcji") as ui:
    gr.Markdown("##Sprawdź transakcję pod kątem oszustwa")

    with gr.Row():
        Gender = gr.Dropdown(["Male", "Female", "Other"], label="Gender")
        Age = gr.Number(label="Age", precision=0)
        State = gr.Textbox(label="State")
        City = gr.Textbox(label="City")
        Account_Type = gr.Dropdown(ACCOUNT_TYPE_CHOICES, label="Account_Type")
        Transaction_Date = gr.Textbox(label="Transaction_Date (YYYY-MM-DD)")
        Transaction_Time = gr.Textbox(label="Transaction_Time (HH:MM:SS)")
    with gr.Row():
        Transaction_Amount = gr.Number(label="Transaction_Amount", precision=2)
        Transaction_Type = gr.Dropdown(TYPE_CHOICES, label="Transaction_Type")
        Merchant_Category = gr.Dropdown(MERCHANT_CATEGORY_CHOICES, label="Merchant_Category")
        Account_Balance = gr.Number(label="Account_Balance", precision=2)
        Transaction_Device = gr.Textbox(label="Transaction_Device")
        Transaction_Location = gr.Textbox(label="Transaction_Location")
        Device_Type = gr.Dropdown(DEVICE_TYPE_CHOICES, label="Device_Type")
        Transaction_Description = gr.Textbox(label="Transaction_Description")

    check_btn = gr.Button("🔍 Sprawdź transakcję")
    verdict_out = gr.Textbox(label="Wynik", interactive=False)
    probas_out = gr.Label(label="Prawdopodobieństwa")

    check_btn.click(
        infer,
        inputs=[
            Gender, Age, State, City, Account_Type, Transaction_Date, Transaction_Time,
            Transaction_Amount, Transaction_Type, Merchant_Category, Account_Balance,
            Transaction_Device, Transaction_Location, Device_Type, Transaction_Description
        ],
        outputs=[verdict_out, probas_out],
    )

app = gr.mount_gradio_app(app, ui, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)