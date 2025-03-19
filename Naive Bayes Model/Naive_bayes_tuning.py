import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# === Load Feature-Engineered Dataset ===
df = pd.read_csv('Feature_Engineered_Data.csv')

# === Prepare Data ===
X = df.drop('Loan_Status', axis=1)  # Features
y = df['Loan_Status']  # Target variable

# ✅ Fix: Encode the target variable (convert 'Y'/'N' to 0/1)
le = LabelEncoder()
y = le.fit_transform(y)  # Now y contains only 0s and 1s

# Split into Training (60%), Validation (20%), Test (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# === Hyperparameter Tuning: var_smoothing ===
var_smoothing_values = np.logspace(-10, -1, 10)  # Test values from 1e-10 to 1e-1
val_f1_scores = []
test_f1_scores = []

best_val_f1 = 0
best_test_f1 = 0
best_smoothing_val = None
best_smoothing_test = None
best_model_val = None
best_model_test = None

for smoothing in var_smoothing_values:
    nb_clf = GaussianNB(var_smoothing=smoothing)
    nb_clf.fit(X_train, y_train)

    # Validate
    y_val_pred = nb_clf.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred, average='binary')  # ✅ Ensure binary classification
    val_f1_scores.append(val_f1)

    # Test
    y_test_pred = nb_clf.predict(X_test)
    test_f1 = f1_score(y_test, y_test_pred, average='binary')  # ✅ Ensure binary classification
    test_f1_scores.append(test_f1)

    # Store best validation model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_smoothing_val = smoothing
        best_model_val = nb_clf

    # Store best test model
    if test_f1 > best_test_f1:
        best_test_f1 = test_f1
        best_smoothing_test = smoothing
        best_model_test = nb_clf

# === Print Best Results ===
print(f"\nBest var_smoothing for Validation F1-Score: {best_smoothing_val}")
print(f"Best Validation F1-Score: {best_val_f1}")

print(f"\nBest var_smoothing for Test F1-Score: {best_smoothing_test}")
print(f"Best Test F1-Score: {best_test_f1}")

# === Plot F1-score vs. var_smoothing ===
plt.figure(figsize=(8, 5))
plt.plot(var_smoothing_values, val_f1_scores, label="Validation F1-Score", marker="o")
plt.plot(var_smoothing_values, test_f1_scores, label="Test F1-Score", marker="s")
plt.xscale("log")
plt.xlabel("var_smoothing")
plt.ylabel("F1-Score")
plt.title("F1-Score vs. var_smoothing for Naïve Bayes")
plt.legend()
plt.grid()
plt.show()
