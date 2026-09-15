# Customer Churn Prediction

A Machine Learning project that predicts customer churn using customer behavior, satisfaction, purchase activity, and other customer-related factors.

## 🌐 Live Demo

[Try the Customer Churn Prediction App](https://customer-churn-prediction-2pmmdedlajtaupappppeuna5.streamlit.app)

## 📌 Project Overview

Customer churn refers to customers who stop using a company's products or services.

The goal of this project is to use Machine Learning to predict whether a customer is likely to churn based on historical customer data.

The project includes:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering
* Machine learning model training
* Model evaluation
* Streamlit deployment

## 🎯 Problem Statement

Businesses need to identify customers who are likely to leave so that they can take preventive actions.

This project builds a machine learning model that predicts customer churn and provides a **churn probability** and **risk category** for a given customer.

## 📊 Dataset

The project uses an **E-Commerce Customer Churn** dataset containing information about customer behavior, satisfaction, purchase activity, and engagement.

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

### Dataset Source

[Kaggle – E-Commerce Customer Churn Analysis and Prediction](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)

## 🔍 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand customer behavior and identify patterns related to churn.

The analysis focused on:

* Customer tenure
* Customer satisfaction
* Order frequency
* Recency of last order
* Customer complaints
* Cashback amount
* Customer activity and engagement

The project also includes feature importance and SHAP visualizations to understand the factors influencing model predictions.

## 🧹 Data Preprocessing

The preprocessing steps include:

* Removing unnecessary columns such as Customer ID
* Handling missing numerical values using median imputation
* Standardizing inconsistent categorical values
* Preparing numerical and categorical features for model training

## ⚙️ Feature Engineering

Additional features were created to capture customer behavior more effectively:

* **Recency Bucket** – groups cust

Developed as a collaborative machine learning project.
