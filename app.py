from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Customer Churn Prediction API")

# --- Load the saved pipeline 
model = joblib.load("model/churn_model.pkl")
raw_input_columns = joblib.load("model/raw_input_columns.pkl")



class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float



def add_engineered_features(df):
    # 1. tenure_group
    def tenure_bucket(months):
        if months <= 12:
            return '0-1 year'
        elif months <= 24:
            return '1-2 years'
        elif months <= 48:
            return '2-4 years'
        else:
            return '4+ years'
    df['tenure_group'] = df['tenure'].apply(tenure_bucket)

    # 2. total_services
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['total_services'] = df[service_cols].apply(lambda row: sum(row == 'Yes'), axis=1)

    # 3. avg_monthly_value (guard divide-by-zero)
    df['avg_monthly_value'] = df['TotalCharges'] / df['tenure'].replace(0, 1)
    return df


@app.get("/")
def home():
    return {"message": "Churn Prediction API. POST to /predict"}


@app.post("/predict")
def predict(customer: Customer):
    data = pd.DataFrame([customer.dict()])

    data = add_engineered_features(data)

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]  

    return {
        "prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(float(probability), 2)
    }