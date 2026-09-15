# Customer Churn Prediction

ML model to predict customer churn using customer behavior, satisfaction, purchase activity, and other customer-related factors.

## 🌐 Live Demo

[Try the Customer Churn Prediction App](https://customer-churn-prediction-2pmmdedlajtaupappppeuna5.streamlit.app)

## 📌 Project Overview

Customer churn refers to customers who stop using a company's products or services.

The goal of this project is to use Machine Learning to predict whether a customer is likely to churn based on historical customer data.

The project includes data preprocessing, exploratory data analysis, feature engineering, machine learning model training, evaluation, and deployment using Streamlit.

## 🎯 Problem Statement

Businesses need to identify customers who are likely to leave so that they can take preventive actions.

This project builds a machine learning model that predicts customer churn and provides a churn probability and risk category for a given customer.

## 📊 Dataset

The project uses an **E-Commerce Customer Churn** dataset containing information about customer behavior and activity.

Important features include:

* Tenure
* Preferred Login Device
* City Tier
* Warehouse to Home Distance
* Preferred Payment Mode
* Satisfaction Score
* Number of Devices Registered
* Preferred Order Category
* Complaints
* Order Count
* Days Since Last Order
* Cashback Amount

Dataset Source:

[Kaggle – E-Commerce Customer Churn Analysis and Prediction](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)

## 🔍 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand customer behavior and identify patterns related to churn.

The analysis focused on factors such as:

* Customer tenure
* Customer satisfaction
* Order frequency
* Recency of last order
* Complaints
* Cashback amount
* Customer activity

## 🧹 Data Preprocessing

The preprocessing steps include:

* Removing unnecessary columns such as Customer ID
* Handling missing values using median imputation
* Standardizing inconsistent categorical values
* Preparing numerical and categorical features for model training

## ⚙️ Feature Engineering

Additional features were created to capture customer behavior more effectively:

* **Recency Bucket** – groups customers based on days since their last order
* **Tenure Bucket** – groups customers based on their tenure
* **Frequent Shopper** – identifies customers with higher order frequency
* **Inactivity Flag** – identifies customers who have been inactive for a longer period
* **Low Satisfaction Flag** – identifies customers with low satisfaction scores
* **High Support Contact Flag** – identifies customers who have submitted complaints
* **Unhappy and Complained** – combines low satisfaction and complaints
* **High Cashback** – identifies customers receiving relatively high cashback

## 🤖 Machine Learning Model

Different machine learning models were evaluated as part of the project.

The deployed application uses an **XGBoost classification model** trained on the E-Commerce customer churn dataset.

The model predicts:

* Churn probability
* Predicted churn class
* Risk category

### Risk Categories

| Churn Probability | Risk        |
| ----------------- | ----------- |
| Below 30%         | Low Risk    |
| 30% – 69%         | Medium Risk |
| 70% and above     | High Risk   |

## 📈 Model Evaluation

The E-Commerce XGBoost model achieved approximately:

* **ROC-AUC:** 0.9996
* **Precision:** 0.9792
* **Recall:** 0.9895
* **F1 Score:** 0.9843

These results were obtained during the model evaluation performed on the dataset.

## 🔑 Factors Influencing Churn

The project focuses on customer behavior factors that can contribute to churn, including:

1. Usage and activity decline
2. Customer tenure
3. Contract/customer relationship characteristics
4. Customer support interactions and complaints
5. Customer satisfaction
6. Pricing/order amount changes
7. Recency and frequency of transactions

## 🌐 Streamlit Application

The trained XGBoost model is integrated into a Streamlit web application.

Users can enter customer information through an interactive form. The application then performs the required feature engineering and passes the processed information to the trained model.

The application displays:

* Churn probability
* Risk level
* Possible risk factors

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Joblib
* Streamlit
* Jupyter Notebook
* Git & GitHub

## 📁 Project Structure

```text
Customer-Churn-Prediction/
│
├── CustomerChurn.ipynb
├── README.md
├── app.py
├── requirements.txt
│
├── models/
│   ├── ECommerce_feature_importance.png
│   ├── ECommerce_shap_summary.png
│   ├── ECommerce_xgb_pipeline.joblib
│   ├── Telco_feature_importance.png
│   ├── Telco_shap_summary.png
│   └── Telco_xgb_pipeline.joblib
│
├── src/
│   ├── feature_engineering.py
│   ├── predict.py
│   └── train.py
│
└── ecommerce_data/
    └── E Commerce Dataset.xlsx
```

## ▶️ How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Baani-Arora/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## 👥 Team

**AI/ML Team Project**

Developed as a collaborative machine learning project.
