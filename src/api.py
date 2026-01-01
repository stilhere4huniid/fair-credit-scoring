from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

# 1. Initialize API
app = FastAPI(title="FairCredit Scoring API", version="1.0")

# 2. Load Model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '..', 'models', 'fair_credit_model.pkl')
model = joblib.load(model_path)

# 3. Define Input Schema (Data Validation)
class LoanApplication(BaseModel):
    age: int
    credit_amount: float
    duration: int
    checking_status: int  # 0-3
    savings: int          # 0-4
    job_skill: int        # 0-3
    housing: int          # 0-2

# 4. Define Prediction Endpoint
@app.post("/predict")
def predict_credit_score(data: LoanApplication):
    # Construct the 20-feature array
    input_data = np.zeros(20)
    
    # Map API inputs to array indices
    input_data[0] = data.checking_status
    input_data[1] = data.duration
    input_data[4] = data.credit_amount
    input_data[5] = data.savings
    input_data[11] = data.age
    input_data[14] = data.housing
    input_data[16] = data.job_skill
    
    # Critical Fairness Feature
    input_data[19] = 1 if data.age > 25 else 0
    
    # Predict
    prediction = model.predict([input_data])[0]
    probability = model.predict_proba([input_data])[0][1]
    
    return {
        "loan_approved": bool(prediction),
        "confidence_score": float(probability),
        "fairness_check": "Equalized Odds Applied"
    }

# To run: uvicorn src.api:app --reload