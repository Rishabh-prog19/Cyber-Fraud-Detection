import joblib
import pandas as pd

# Load Saved Model
model = joblib.load("fraud_model.pkl")

print("Model Loaded Successfully")

# Load Dataset (for demo prediction)
df = pd.read_csv("data/creditcard.csv")

# Take one random transaction
sample = df.drop("Class", axis=1).iloc[[0]]

prediction = model.predict(sample)

if prediction[0] == 1:
    print("Fraud Transaction Detected")
else:
    print("Normal Transaction")