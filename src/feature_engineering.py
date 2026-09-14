import pandas as pd
import numpy as np
import os

def load_and_prep_telco(data_path="telco_data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    df = pd.read_csv(data_path)
    # Baani's prep
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df["tenure_bucket"] = pd.cut(df["tenure"], bins=[0,12,24,48,72], labels=["0-12mo","13-24mo","25-48mo","49-72mo"])
    
    # Advanced Feature Engineering for Telco
    # Usage decline proxy / Pricing features
    # ratio of total charges to tenure (proxy for average monthly charge). Handle div by zero if tenure is 0.
    df["AvgCharge_vs_Monthly"] = np.where(df["tenure"] > 0, (df["TotalCharges"] / df["tenure"]) - df["MonthlyCharges"], 0)
    df["high_monthly_charge"] = (df["MonthlyCharges"] > df["MonthlyCharges"].quantile(0.75)).astype(int)
    
    # Contract type & Customer support interactions
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["has_tech_support"] = (df["TechSupport"] == "Yes").astype(int)
    
    # Interactions
    df["high_risk_contract_no_support"] = (df["is_month_to_month"] & (df["has_tech_support"] == 0)).astype(int)
    
    # Create simple categories for prediction UI later
    df["Support_Interactions_Level"] = df["TechSupport"].replace({"No": "Low", "No internet service": "None", "Yes": "High"})
    
    return df

def load_and_prep_ecom(data_path="ecommerce_data/E Commerce Dataset.xlsx"):
    df = pd.read_excel(data_path, sheet_name="E Comm")
    # Baani's prep
    df = df.drop(columns=["CustomerID"])
    numeric_cols_with_nulls = ["Tenure", "WarehouseToHome", "HourSpendOnApp", "OrderAmountHikeFromlastYear", "CouponUsed", "OrderCount", "DaySinceLastOrder"]
    for col in numeric_cols_with_nulls:
        df[col] = df[col].fillna(df[col].median())
        
    df["PreferredLoginDevice"] = df["PreferredLoginDevice"].replace({"Phone": "Mobile Phone"})
    df["PreferredPaymentMode"] = df["PreferredPaymentMode"].replace({"CC": "Credit Card", "COD": "Cash on Delivery"})
    df["PreferedOrderCat"] = df["PreferedOrderCat"].replace({"Mobile": "Mobile Phone"})
    
    df["recency_bucket"] = pd.cut(df["DaySinceLastOrder"], bins=[-1,3,7,15,50], labels=["0-3d","4-7d","8-15d","16d+"])
    df["tenure_bucket"] = pd.cut(df["Tenure"], bins=[-1,3,6,12,24,60], labels=["0-3mo","4-6mo","7-12mo","13-24mo","25mo+"])
    
    # Advanced Feature Engineering for E-Commerce
    # Usage / Recency
    df["frequent_shopper"] = (df["OrderCount"] > df["OrderCount"].median()).astype(int)
    df["inactivity_flag"] = (df["DaySinceLastOrder"] > 15).astype(int)
    
    # Satisfaction and Complaints
    df["low_satisfaction_flag"] = (df["SatisfactionScore"] <= 2).astype(int)
    df["high_support_contact_flag"] = df["Complain"] # already 0/1
    df["unhappy_and_complained"] = (df["low_satisfaction_flag"] & df["high_support_contact_flag"]).astype(int)
    
    # Pricing
    df["high_cashback"] = (df["CashbackAmount"] > df["CashbackAmount"].quantile(0.75)).astype(int)
    
    return df

if __name__ == "__main__":
    t_df = load_and_prep_telco()
    print("Telco prep shape:", t_df.shape)
    e_df = load_and_prep_ecom()
    print("E-com prep shape:", e_df.shape)
