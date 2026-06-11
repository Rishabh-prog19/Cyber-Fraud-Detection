import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Load Dataset
df = pd.read_csv("data/creditcard.csv")

print("Dataset Shape:", df.shape)

# Scale Amount
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])

# Split
X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("After SMOTE:", y_train_res.value_counts())

# RANDOM FOREST
model = RandomForestClassifier(
    n_estimators=100,
    random_state=100,
    n_jobs=-1
)

model.fit(X_train_res, y_train_res)

print("Model Training Completed")

# Prediction
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

roc = roc_auc_score(y_test, y_pred)
print("ROC AUC:", roc)
import matplotlib.pyplot as plt
import numpy as np

importances = model.feature_importances_
indices = np.argsort(importances)[-10:]

plt.figure(figsize=(10,6))
plt.title("Top 10 Important Features")
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), X.columns[indices])
plt.xlabel("Feature Importance")
plt.show()

import joblib

joblib.dump(model, "fraud_model.pkl")

print("Model Saved Successfully")