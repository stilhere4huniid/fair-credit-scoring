# 📘 Technical Methodology & Bias Audit Report

This document details the end-to-end data science workflow used to build the **FairCredit** engine, from initial Exploratory Data Analysis (EDA) to the final deployment of the fairness-constrained XGBoost model.

---

## 1. Data Loading & Preprocessing
**Source:** UCI German Credit Dataset (1,000 samples, 20 features).

### 🚨 The "Zero-Bias" Discovery
During EDA, we discovered a critical data quality issue in the financial history features (`checking_status` and `savings`).
* **Issue:** Missing values were implicitly encoded as `0`.
* **Impact:** The model interpreted young applicants with "no checking account" as having "debt" or "zero savings," leading to unfair rejections.
* **Fix:** We implemented a **Custom Imputation Strategy** to differentiate between "No Account" (Neutral) and "Zero Balance" (Risky), injecting domain-specific "Safe" defaults for the web interface.

---

## 2. Baseline Model Performance (The "Biased" Model)
We first trained a standard **XGBoost Classifier** without any fairness constraints to establish a baseline.

* **Accuracy:** 78.50%
* **Bias Metric:** We measured **Demographic Parity Difference** (Selection Rate Gap) between:
    * **Group 0 (Protected):** Age < 25
    * **Group 1 (Privileged):** Age ≥ 25
* **Result:** The baseline model favored older applicants with a **Bias Gap of 11.58%**. This violated standard fair lending thresholds (typically < 5%).

---

## 3. Mitigation Strategy: The "Fairness Grid"
To reduce this gap, we did *not* simply remove the "Age" variable (which fails due to proxy variables). Instead, we used **Microsoft Fairlearn** to apply algorithmic constraints.

### The Algorithm: Exponentiated Gradient Reduction
We formulated the optimization problem as:
$$\min \text{Error}(h) \quad \text{s.t.} \quad \text{FairnessConstraint}(h) \le \epsilon$$

### The Grid Search
We trained **20 candidate models** along the Pareto Frontier, trading off a small amount of accuracy for a massive gain in fairness.

| Model ID | Accuracy | Bias Gap (Diff) | Decision |
| :--- | :--- | :--- | :--- |
| Model 0 | 70.50% | 84.28% | ❌ Too Biased |
| Model 10 | 79.00% | 11.58% | ❌ Baseline |
| **Model 14** | **81.50%** | **2.58%** | **✅ SELECTED** |
| Model 19 | 77.00% | 0.05% | ❌ Accuracy Loss |

**Winner (Model 14):**
* **Accuracy:** 81.5% (+2.5% vs Baseline)
* **Bias Gap:** 2.58% (Reduced by ~78%)
* **Conclusion:** This model achieved the "Goldilocks" zone—compliant with regulations (<5% gap) while actually *improving* accuracy by regularizing the decision boundary.

---

## 4. Post-Processing & Explainability
### SHAP Waterfall Analysis
To comply with GDPR "Right to Explanation," we integrated **SHAP (Shapley Additive Explanations)**.
* Global Feature Importance confirmed that `Checking Status` and `Savings` are the dominant predictors.
* Local Explainability allows us to generate a specific "Why?" chart for every rejected applicant.

---

## 5. Deployment Architecture
The final model was serialized using `joblib` and deployed via a decoupled microservice architecture:
1.  **Frontend:** Streamlit (User Interface & Scenario Testing).
2.  **Backend:** FastAPI (Prediction Endpoint & Validation).
3.  **Containerization:** (Roadmap) Docker support for cloud scaling.