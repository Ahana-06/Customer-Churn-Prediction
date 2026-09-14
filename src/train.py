import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap

def build_and_evaluate(df, target_col, name, output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*40}\nProcessing {name} Dataset\n{'='*40}")
    
    # Check class imbalance
    churn_count = df[target_col].sum()
    total = len(df)
    print(f"Total samples: {total}")
    print(f"Churn count: {churn_count} ({churn_count/total:.2%})")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Identify column types
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)
        ]
    )
    
    # Imbalance strategy
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    
    # 1. Logistic Regression Baseline
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_preds = lr_pipeline.predict(X_test)
    lr_probs = lr_pipeline.predict_proba(X_test)[:, 1]
    
    print("\n[Logistic Regression Results]")
    print(f"ROC-AUC: {roc_auc_score(y_test, lr_probs):.4f}")
    print(f"Precision: {precision_score(y_test, lr_preds):.4f}")
    print(f"Recall: {recall_score(y_test, lr_preds):.4f}")
    print(f"F1: {f1_score(y_test, lr_preds):.4f}")
    
    # 2. XGBoost
    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        ))
    ])
    xgb_pipeline.fit(X_train, y_train)
    xgb_preds = xgb_pipeline.predict(X_test)
    xgb_probs = xgb_pipeline.predict_proba(X_test)[:, 1]
    
    print("\n[XGBoost Results]")
    print(f"ROC-AUC: {roc_auc_score(y_test, xgb_probs):.4f}")
    print(f"Precision: {precision_score(y_test, xgb_preds):.4f}")
    print(f"Recall: {recall_score(y_test, xgb_preds):.4f}")
    print(f"F1: {f1_score(y_test, xgb_preds):.4f}")
    print("\nConfusion Matrix (XGBoost):")
    print(confusion_matrix(y_test, xgb_preds))
    print("\nClassification Report (XGBoost):")
    print(classification_report(y_test, xgb_preds))
    
    # Save the XGBoost pipeline
    joblib.dump(xgb_pipeline, os.path.join(output_dir, f"{name}_xgb_pipeline.joblib"))
    print(f"\nSaved {name} model artifact.")
    
    # Feature Importance & SHAP
    # Get feature names after preprocessing
    cat_encoder = preprocessor.named_transformers_['cat']
    if len(categorical_cols) > 0:
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols)
    else:
        cat_feature_names = []
    feature_names = numeric_cols + list(cat_feature_names)
    
    xgb_model = xgb_pipeline.named_steps['classifier']
    importances = xgb_model.feature_importances_
    
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    fi_df = fi_df.sort_values(by='Importance', ascending=False).head(15)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=fi_df, x='Importance', y='Feature', palette='viridis')
    plt.title(f"{name} - Top 15 XGBoost Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_feature_importance.png"))
    plt.close()
    
    # SHAP Explainer
    X_train_transformed = preprocessor.transform(X_train)
    # convert to dense array or dataframe for SHAP
    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()
        
    X_test_transformed = preprocessor.transform(X_test)
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()
        
    # We use a subset for shap to save time
    shap_sample = shap.sample(X_train_transformed, 200)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test_transformed[:500]) # only top 500 for testing
    
    plt.figure()
    shap.summary_plot(shap_values, X_test_transformed[:500], feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_shap_summary.png"))
    plt.close()
    print(f"Generated and saved explainability plots for {name}.")

if __name__ == "__main__":
    from feature_engineering import load_and_prep_telco, load_and_prep_ecom
    
    # Telco
    telco_df = load_and_prep_telco("telco_data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    build_and_evaluate(telco_df, target_col="Churn", name="Telco", output_dir="models")
    
    # E-commerce
    ecom_df = load_and_prep_ecom("ecommerce_data/E Commerce Dataset.xlsx")
    build_and_evaluate(ecom_df, target_col="Churn", name="ECommerce", output_dir="models")
