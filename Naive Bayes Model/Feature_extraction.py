import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# === Load the dataset ===
df = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')

# === Data Cleaning (Ensure No Missing Values) ===
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Fill missing numeric values with median
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill missing categorical values with mode
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# === Fix: Convert "Dependents" Column Properly ===
df['Dependents'] = df['Dependents'].replace({'3+': '3'}).astype(int)

# === Feature Engineering ===
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['IncomeLoanRatio'] = df['TotalIncome'] / (df['LoanAmount'] + 1)  # Avoid division by zero
df['Loan_Amount_Term_Years'] = df['Loan_Amount_Term'] / 12
df['Good_Credit_History'] = (df['Credit_History'] == 1).astype(int)
df['Log_TotalIncome'] = np.log(df['TotalIncome'] + 1)
df['Log_LoanAmount'] = np.log(df['LoanAmount'] + 1)

# Drop Loan_ID
df.drop(columns=['Loan_ID'], inplace=True)

# === Normalization: Scale Numeric Features ===
scaler = StandardScaler()
numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                    'TotalIncome', 'IncomeLoanRatio', 'Loan_Amount_Term_Years', 
                    'Log_TotalIncome', 'Log_LoanAmount']
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# One-Hot Encode Categorical Features
df = pd.get_dummies(df, columns=['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area'], drop_first=True)

# Save the engineered dataset for model retraining
df.to_csv('Feature_Engineered_Data.csv', index=False)

print("\nFeature Engineering Completed. Dataset saved as 'Feature_Engineered_Data.csv'")
