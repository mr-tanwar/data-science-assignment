# Customer Churn Prediction

This project is an end-to-end machine learning solution for predicting whether a telecom customer is likely to churn, or cancel their service.

The goal is to help the retention team identify customers who may be at risk of leaving, so they can take action before that happens.

The project covers the complete workflow, including data cleaning, exploratory data analysis, feature engineering, model training, evaluation, model saving, and serving predictions through a REST API.

## Business Problem

Customer churn is an important problem for telecom companies because losing an existing customer usually costs more than retaining one.

Using the IBM Telco Customer Churn dataset, this project builds a Decision Tree model to identify customers who are more likely to churn.
Column meanings were confirmed using the provided data dictionary.

The trained model is then exposed through a FastAPI REST API so predictions can also be made for new customer data.

## Project Structure

```text
customer_churn_project/
├── data/
│   └── Telco customer churn dataset
├── notebook/
│   └── churn_analysis.ipynb
├── model/
│   ├── churn_model.pkl
│   └── raw_input_columns.pkl
├── app.py
├── requirements.txt
├── README.md
└── sample_request.json
```

The main files are:

- `data/` — contains the customer churn dataset.
- `notebook/churn_analysis.ipynb` — contains the complete analysis, EDA, feature engineering, model training, and evaluation.
- `model/churn_model.pkl` — saved preprocessing and machine learning pipeline.
- `model/raw_input_columns.pkl` — stores the expected input columns used by the API.
- `app.py` — FastAPI application used to serve predictions.
- `requirements.txt` — list of required Python packages.
- `sample_request.json` — example request that can be sent to the API.

## Setup and Installation

Repo link : https://github.com/mr-tanwar/data-science-assignment

Make sure Python 3.10 or later is installed.

From the project root, install the required packages using:

```bash
pip install -r requirements.txt
```

## Running the Notebook

Open:

```text
notebook/churn_analysis.ipynb
```

in Jupyter Notebook, JupyterLab, or Google Colab.

Run the cells from top to bottom.

The notebook covers the full machine learning workflow, including:

- Data understanding and cleaning
- Exploratory data analysis
- Feature engineering
- Data preprocessing
- Model training
- Model comparison
- Evaluation
- Model interpretation
- Saving the final pipeline

## Running the API

From the project root, start the FastAPI server using:

```bash
uvicorn app:app --reload
```

Once the server starts, the API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI also provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

The `/docs` page can be used to test the prediction endpoint directly from the browser.

## API Endpoint

### POST `/predict`

The prediction endpoint accepts customer information in JSON format.

It applies the same preprocessing and feature engineering steps that were used during model training and then passes the processed data to the trained model.

The response includes both the predicted churn result and the estimated churn probability.

If required fields are missing or invalid values are provided, the API returns a validation error with details about the problem.

## Sample Request

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.35,
  "TotalCharges": 140.7
}
```

## Sample Response

```json
{
  "prediction": "Yes",
  "churn_probability": 0.53
}
```

## Approach

The first step was to clean and prepare the dataset.

`TotalCharges` was initially stored as text because a small number of rows contained blank values. These rows belonged to new customers with zero tenure, so their `TotalCharges` values were replaced with `0`.

The `customerID` column was removed because it is only an identifier and does not provide useful information for churn prediction.

Categorical variables were converted using one-hot encoding, while numerical variables were scaled. All preprocessing steps were included inside a scikit-learn pipeline so the same transformations are applied consistently during both training and prediction.

The dataset was split into 70% training data and 30% test data using `random_state=42`. Stratified sampling was used so both datasets maintained approximately the same churn distribution as the original data.

Feature engineering was also used to provide the model with additional customer information.

A `TenureGroup` feature was created to group customers based on how long they had been with the company. A `TotalServices` feature was also created to represent the number of services used by each customer.

Two Decision Tree configurations were compared.

The unrestricted tree achieved almost perfect performance on the training data but performed much worse on the test data, showing clear signs of overfitting.

A depth-limited Decision Tree was therefore selected as the final model because it generalized better to unseen customer data.

## Key Findings

Contract type was one of the strongest factors associated with customer churn. Customers on month-to-month contracts showed much higher churn than customers on longer-term contracts.

Customer tenure was also important. New customers were more likely to churn, especially during the early stages of their relationship with the company.

Fiber-optic customers also showed higher churn compared with customers using DSL or no internet service.

Overall, the model identified a common higher-risk profile as a relatively new customer who is on a month-to-month contract and uses fiber-optic internet.

The final model identifies around 60% of customers who actually churn.

## Why Recall Matters

For this problem, Recall is more important than relying only on overall accuracy.

A false negative happens when the model predicts that a customer will stay, but the customer actually churns. This is costly because the company loses the opportunity to intervene before the customer leaves.

A false positive means the model identifies a customer as being at risk even though they would have stayed. In that case, the company may only spend a small amount on a retention action, such as an offer, email, or customer-service call.

Because missing an actual churner is usually more costly than contacting a customer who was not going to leave, the model places more importance on identifying as many real churners as possible.

Accuracy alone can also be misleading because the dataset is imbalanced. Around 73% of customers do not churn, so a model that simply predicts "No Churn" for every customer could still achieve around 73% accuracy while failing to identify any real churners.
