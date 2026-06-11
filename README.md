# Cyber Fraud Prediction with Advanced Machine Learning

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. The dataset contains highly imbalanced transaction records, making fraud detection a challenging classification problem.

To address this issue, SMOTE (Synthetic Minority Oversampling Technique) was used to balance the classes before training a Random Forest model. The trained model can then classify incoming transactions as either legitimate or fraudulent.

---

## Problem Statement

Financial fraud causes significant losses every year. Since fraudulent transactions represent only a small percentage of all transactions, traditional classification methods often struggle to identify them accurately.

The objective of this project is to build a machine learning model that can effectively distinguish fraudulent transactions from normal ones while minimizing false predictions.

---

## Dataset

The project uses a credit card transaction dataset containing:

- Transaction features
- Transaction amount
- Class label
  - 0 = Normal Transaction
  - 1 = Fraudulent Transaction

Due to privacy concerns, most features are anonymized.
The dataset is not included in this repository because of GitHub's file size limitations.

Download it from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the downloaded creditcard.csv file inside the data/ directory before running the project.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Imbalanced-Learn (SMOTE)
- Pickle

---

## Project Structure


Cyber-Fraud-Prediction/
│
├── data/
│   └── creditcard.csv
│
├── src/
│   ├── train_model.py
│   ├── predict.py
│   └── app.py
│
├── fraud_model.pkl
├── requirements.txt
└── README.md


---

## Methodology

# 1. Data Preprocessing

- Loaded transaction data using Pandas
- Standardized transaction amounts using StandardScaler
- Split the dataset into training and testing sets

# 2. Handling Class Imbalance

Since fraudulent transactions are very rare compared to legitimate ones, SMOTE was applied to generate synthetic fraud samples and improve model learning.

# 3. Model Training

A Random Forest Classifier was trained on the balanced dataset.

Reasons for choosing Random Forest:

- Handles large datasets effectively
- Works well on classification tasks
- Reduces overfitting through ensemble learning
- Provides strong performance with minimal parameter tuning

# 4. Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score

---

## Running the Project

### Clone the Repository


git clone https://github.com/your-username/Cyber-Fraud-Prediction.git
cd Cyber-Fraud-Prediction


### Install Dependencies


pip install -r requirements.txt


### Train the Model


python src/train_model.py


### Run Prediction

python src/predict.py


---

## Sample Output


Model Loaded Successfully

Normal Transaction


or


Model Loaded Successfully

Fraudulent Transaction Detected


---

## Future Improvements

Some areas that can be explored in future versions:

- Hyperparameter tuning
- Feature engineering
- XGBoost and LightGBM comparison
- Real-time fraud detection pipeline
- Streamlit web interface
- Model deployment using cloud services

---

## Learning Outcomes

Through this project, I gained practical experience with:

- Data preprocessing
- Handling imbalanced datasets
- Machine learning model development
- Model evaluation metrics
- Saving and loading trained models
- Building an end-to-end fraud detection workflow

---

## Author

Rishabh Shrivastava
