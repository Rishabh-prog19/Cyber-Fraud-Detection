import streamlit as st
import joblib
import pandas as pd
import os


st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection")


model = None

try:
    if os.path.exists("fraud_model.pkl"):
        model = joblib.load("fraud_model.pkl")
    elif os.path.exists("src/fraud_model.pkl"):
        model = joblib.load("src/fraud_model.pkl")
    elif os.path.exists("../fraud_model.pkl"):
        model = joblib.load("../fraud_model.pkl")

    if model is not None:
        st.success("Model loaded successfully !!")
    else:
        st.error("Model file not found !!")

except Exception as e:
    st.error(f"Error loading model: {e}")

# Load Dataset
# -----------------------------
try:
    if os.path.exists("data/creditcard.csv"):
        df = pd.read_csv("data/creditcard.csv")
    elif os.path.exists("../data/creditcard.csv"):
        df = pd.read_csv("../data/creditcard.csv")
    else:
        df = None
        st.error("Dataset not found ❌")

except Exception as e:
    df = None
    st.error(f"Error loading dataset: {e}")

# -----------------------------
# Mode Selection
# -----------------------------
st.subheader("Choose Input Mode")

mode = st.radio(
    "Select how you want to test:",
    ["📊 Use Sample Transaction", "🎯 Manual Input (Simplified)"]
)

# -----------------------------
# SAMPLE MODE (BEST)
# -----------------------------
if mode == "📊 Use Sample Transaction":

    st.write("Click below to test a real transaction from dataset")

    if st.button("Generate Sample"):

        if df is None or model is None:
            st.error("Missing model or dataset ❌")
        else:
            sample = df.sample(1)

            st.write("### Sample Transaction")
            st.dataframe(sample)

            X_sample = sample.drop("Class", axis=1)

            prediction = model.predict(X_sample)[0]

            prob = None
            try:
                prob = model.predict_proba(X_sample)[0][1]
            except:
                pass

            if prediction == 1:
                st.error("🚨 Fraudulent Transaction Detected!")
            else:
                st.success("✅ Normal Transaction")

            if prob is not None:
                st.info(f"Fraud Probability: {round(prob * 100, 2)}%")

# -----------------------------
# MANUAL MODE (SIMPLIFIED)
# -----------------------------
else:
    st.write("Enter limited features (demo purpose only)")

    amount = st.number_input("Transaction Amount", min_value=0.0)
    v1 = st.number_input("Feature V1")
    v2 = st.number_input("Feature V2")

    if st.button("Predict"):

        if model is None:
            st.error("Model not loaded ❌")
        else:
            try:
                # NOTE: This is approximate input
                input_data = [[0, v1, v2] + [0]*26 + [amount]]

                prediction = model.predict(input_data)[0]

                if prediction == 1:
                    st.error("🚨 Fraud Detected")
                else:
                    st.success("✅ Normal Transaction")

                st.warning("⚠️ This mode uses partial inputs, results may be less accurate.")

            except Exception as e:
                st.error(f"Prediction error: {e}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("🔍 Model trained using Random Forest with SMOTE and PCA-transformed features.")