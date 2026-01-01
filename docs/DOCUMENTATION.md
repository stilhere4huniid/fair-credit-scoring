# 📖 System Documentation & User Guide

This document serves as the operational manual for the **FairCredit** system. It details the user interface features, the logic behind the simulation scenarios, and the API specification.

---

## 🖥️ 1. The Streamlit Dashboard (Digital Boardroom)
The dashboard is designed for **Model Risk Management (MRM)** officers to audit the model's decisions in real-time.

### A. The Input Sidebar
* **Demographics:**
    * **Age:** Primary protected variable. The system flags inputs `< 25` as "Protected Group: YOUTH".
    * **Sex:** Included for completeness but *excluded* from the training constraints (Fairness is optimized on Age only).
* **Financial Health Toggle (Simulation Control):**
    * *Note:* This dropdown is a simulation tool for the demo. In a production environment, this data would come directly from a SQL database.
    * **"Wealthy":** Injects `Checking > 200 DM` and `Savings > 1000 DM`. (Used to test Scenario A).
    * **"Average":** Injects `Checking 0-200 DM` and `Savings 100-500 DM`.
    * **"Poor":** Injects `Checking < 0 DM` and `No Savings`. (Used to test Scenario B).

### B. Interpreting the Decision Engine
* **✅ LOAN APPROVED (Green):**
    * The model predicts the applicant will repay the loan.
    * **Confidence Score:** The probability of repayment (e.g., 80.2%).
* **❌ LOAN REJECTED (Red):**
    * The model predicts a default.
    * **Risk Score:** The probability of default (1 - Confidence).

### C. Explainability (SHAP Waterfall)
The "Why?" section satisfies **GDPR Article 22** (Right to Explanation).
* **Red Bars (+):** Features pushing the score *towards* approval (e.g., High Savings).
* **Blue Bars (-):** Features pushing the score *towards* rejection (e.g., Unskilled Job).

---

## 🔌 2. API Reference (Microservice)
The model is exposed via a REST API using **FastAPI**.

### Endpoint: `POST /predict`
* **URL:** `http://localhost:8000/predict`
* **Content-Type:** `application/json`

### Request Body Schema
```json
{
  "age": 21,                // Integer: Applicant age
  "credit_amount": 1500,    // Float: Amount in Deutsche Marks
  "duration": 12,           // Integer: Loan duration in months
  "checking_status": 2,     // Integer: 0 (<0), 1 (0-200), 2 (>200), 3 (None)
  "savings": 3,             // Integer: 0 (<100), 1 (100-500), 2 (500-1k), 3 (>1k)
  "job_skill": 2,           // Integer: 0 (Unskilled), 1 (Resident), 2 (Skilled), 3 (Mgmt)
  "housing": 2              // Integer: 0 (Free), 1 (Rent), 2 (Own)
}
```

### Response Schema
```json
{
  "loan_approved": true,                  // Boolean: Final Decision
  "confidence_score": 0.812,              // Float: 0.0 to 1.0
  "fairness_check": "Equalized Odds Applied" // String: Governance Tag
}
```

---

## 🔧 3. Error Codes & Troubleshooting

| Error | Cause | Fix |
| :--- | :--- | :--- |
| **422 Validation Error** | Sending text instead of numbers in JSON. | Ensure all inputs match the Schema types (Integers/Floats). |
| **500 Internal Error** | Model file missing or corrupted. | Check if `models/fair_credit_model.pkl` exists. |
| **Streamlit "Zero" Error** | Feature name mismatch. | Ensure `src/app.py` feature list matches the training notebook. |

---

## 🔄 4. Retraining Protocol
To update the model with new data:
1.  Place new data in `data/raw/`.
2.  Run the `notebooks/fairness_audit.ipynb` Jupyter notebook.
3.  The notebook will automatically save the new model to `models/fair_credit_model.pkl`.
4.  Restart the API server to load the new weights.
