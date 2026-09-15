import streamlit as st
import pandas as pd
import joblib


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(
        "models/ECommerce_xgb_pipeline.joblib"
    )


model = load_model()


# ---------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------

def create_features(df):

    # Recency bucket
    df["recency_bucket"] = pd.cut(
        df["DaySinceLastOrder"],
        bins=[-1, 3, 7, 15, 50],
        labels=["0-3d", "4-7d", "8-15d", "16d+"]
    )

    # Tenure bucket
    df["tenure_bucket"] = pd.cut(
        df["Tenure"],
        bins=[-1, 3, 6, 12, 24, 60],
        labels=[
            "0-3mo",
            "4-6mo",
            "7-12mo",
            "13-24mo",
            "25mo+"
        ]
    )

    # Frequent shopper
    df["frequent_shopper"] = (
        df["OrderCount"] > 2.0
    ).astype(int)

    # Inactivity flag
    df["inactivity_flag"] = (
        df["DaySinceLastOrder"] > 15
    ).astype(int)

    # Low satisfaction
    df["low_satisfaction_flag"] = (
        df["SatisfactionScore"] <= 2
    ).astype(int)

    # High support contact
    df["high_support_contact_flag"] = df["Complain"]

    # Unhappy and complained
    df["unhappy_and_complained"] = (
        (df["low_satisfaction_flag"] == 1) &
        (df["high_support_contact_flag"] == 1)
    ).astype(int)

    # High cashback
    df["high_cashback"] = (
        df["CashbackAmount"] > 196.3925
    ).astype(int)

    return df


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

st.title("📊 Customer Churn Prediction")

st.write(
    "Enter key customer details to estimate the probability "
    "that the customer may churn."
)

st.divider()


# ---------------------------------------------------------
# Customer Information
# ---------------------------------------------------------

st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=60,
        value=10
    )

    city_tier = st.selectbox(
        "City Tier",
        [1, 2, 3]
    )

    satisfaction = st.slider(
        "Satisfaction Score",
        min_value=1,
        max_value=5,
        value=3
    )


with col2:

    complain = st.selectbox(
        "Has the customer complained?",
        ["No", "Yes"]
    )

    order_count = st.number_input(
        "Order Count",
        min_value=0,
        max_value=100,
        value=2
    )

    days_since_last_order = st.number_input(
        "Days Since Last Order",
        min_value=0,
        max_value=50,
        value=5
    )


with col3:

    hours_on_app = st.number_input(
        "Hours Spent on App",
        min_value=0.0,
        max_value=10.0,
        value=2.0
    )

    cashback = st.number_input(
        "Cashback Amount",
        min_value=0.0,
        max_value=500.0,
        value=150.0
    )

    order_category = st.selectbox(
        "Preferred Order Category",
        [
            "Laptop & Accessory",
            "Mobile Phone",
            "Fashion",
            "Grocery",
            "Others"
        ]
    )


# ---------------------------------------------------------
# Purchase Preference
# ---------------------------------------------------------

st.subheader("🛒 Purchase Preference")

payment_mode = st.selectbox(
    "Preferred Payment Mode",
    [
        "Credit Card",
        "Cash on Delivery",
        "Debit Card",
        "UPI",
        "E wallet"
    ]
)


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

st.divider()

if st.button(
    "🔮 Predict Churn Risk",
    use_container_width=True
):

    complain_value = 1 if complain == "Yes" else 0

    # -----------------------------------------------------
    # Hidden / Default Values
    # These fields are still required by the trained model.
    # -----------------------------------------------------

    customer = pd.DataFrame([{

        # User inputs
        "Tenure": tenure,

        # Default values
        "PreferredLoginDevice": "Mobile Phone",

        "CityTier": city_tier,

        "WarehouseToHome": 15.0,

        "PreferredPaymentMode": payment_mode,

        "Gender": "Male",

        "HourSpendOnApp": hours_on_app,

        "NumberOfDeviceRegistered": 3,

        "PreferedOrderCat": order_category,

        "SatisfactionScore": satisfaction,

        "MaritalStatus": "Single",

        "NumberOfAddress": 2,

        "Complain": complain_value,

        "OrderAmountHikeFromlastYear": 15.0,

        "CouponUsed": 2,

        "OrderCount": order_count,

        "DaySinceLastOrder": days_since_last_order,

        "CashbackAmount": cashback
    }])


    # -----------------------------------------------------
    # Add Engineered Features
    # -----------------------------------------------------

    customer = create_features(customer)


    # -----------------------------------------------------
    # Model Prediction
    # -----------------------------------------------------

    probability = model.predict_proba(customer)[0][1]

    prediction = model.predict(customer)[0]

    probability_percent = probability * 100


    # -----------------------------------------------------
    # Risk Classification
    # -----------------------------------------------------

    if probability < 0.30:

        risk = "Low Risk"

    elif probability < 0.70:

        risk = "Medium Risk"

    else:

        risk = "High Risk"


    # -----------------------------------------------------
    # Prediction Results
    # -----------------------------------------------------

    st.subheader("📈 Prediction Result")

    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.metric(
            "Churn Probability",
            f"{probability_percent:.2f}%"
        )


    with result_col2:

        st.metric(
            "Risk Level",
            risk
        )


    if risk == "High Risk":

        st.error(
            "⚠️ This customer has a high probability of churning."
        )

    elif risk == "Medium Risk":

        st.warning(
            "⚠️ This customer has a moderate churn risk."
        )

    else:

        st.success(
            "✅ This customer has a low churn risk."
        )


    # -----------------------------------------------------
    # Possible Risk Factors
    # -----------------------------------------------------

    st.subheader("🔑 Possible Risk Factors")

    factors = []


    if days_since_last_order > 15:

        factors.append(
            "Customer has been inactive for a long time."
        )


    if satisfaction <= 2:

        factors.append(
            "Low customer satisfaction."
        )


    if complain_value == 1:

        factors.append(
            "Customer has reported a complaint."
        )


    if tenure <= 3:

        factors.append(
            "Customer has a short tenure."
        )


    if order_count <= 2:

        factors.append(
            "Customer has relatively low order frequency."
        )


    if hours_on_app < 1:

        factors.append(
            "Low app engagement."
        )


    if not factors:

        factors.append(
            "No major risk factors identified "
            "from the provided inputs."
        )


    for factor in factors:

        st.write("•", factor)


# ---------------------------------------------------------
# Model Performance Comparison
# ---------------------------------------------------------

st.divider()

st.subheader(
    "📊 Feature Reduction: 10 Features → 5 Features"
)

st.write(
    "The following comparison shows how reducing the number "
    "of features affected the performance of the models."
)


# ---------------------------------------------------------
# Telco Dataset
# ---------------------------------------------------------

st.markdown("### 📞 Telco Customer Churn")

telco_col1, telco_col2, telco_col3 = st.columns(3)


with telco_col1:

    st.metric(
        "ROC-AUC",
        "79.9%",
        delta="-4.5%",
        delta_color="inverse"
    )


with telco_col2:

    st.metric(
        "PR-AUC",
        "51.2%",
        delta="-14.2%",
        delta_color="inverse"
    )


with telco_col3:

    st.metric(
        "F1 Score",
        "0.56",
        delta="-0.03",
        delta_color="inverse"
    )


st.caption(
    "Original → Reduced: "
    "ROC-AUC 84.4% → 79.9% | "
    "PR-AUC 65.4% → 51.2% | "
    "F1 0.59 → 0.56"
)


# ---------------------------------------------------------
# E-Commerce Dataset
# ---------------------------------------------------------

st.markdown("### 🛒 E-Commerce Customer Churn")

ecom_col1, ecom_col2, ecom_col3 = st.columns(3)


with ecom_col1:

    st.metric(
        "ROC-AUC",
        "86.9%",
        delta="-8.3%",
        delta_color="inverse"
    )


with ecom_col2:

    st.metric(
        "PR-AUC",
        "57.6%",
        delta="-23.5%",
        delta_color="inverse"
    )


with ecom_col3:

    st.metric(
        "F1 Score",
        "0.59",
        delta="-0.14",
        delta_color="inverse"
    )


st.caption(
    "Original → Reduced: "
    "ROC-AUC 95.2% → 86.9% | "
    "PR-AUC 81.1% → 57.6% | "
    "F1 0.73 → 0.59"
)


# ---------------------------------------------------------
# Dataset Sources
# ---------------------------------------------------------

st.divider()

st.subheader("📁 Dataset Sources")

st.markdown(
    """
    - **Telco Customer Churn:**  
      https://www.kaggle.com/datasets/blastchar/telco-customer-churn

    - **E-Commerce Customer Churn:**  
      https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction
    """
)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "Customer Churn Prediction | E-Commerce XGBoost Model"
)