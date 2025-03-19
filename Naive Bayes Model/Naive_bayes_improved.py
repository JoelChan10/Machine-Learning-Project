import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# === Load the Feature-Engineered Dataset ===
df = pd.read_csv('Feature_Engineered_Data.csv')  # Load the dataset with new features

# === Prepare the Data for Modeling ===
X = df.drop('Loan_Status', axis=1)  # Use all features except target
y = df['Loan_Status']

# Encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data: 60% training, 20% validation, 20% test.
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

print("\nDataset splits:")
print("Training set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_val.shape, y_val.shape)
print("Test set shape:", X_test.shape, y_test.shape)

# === Improved Model Training: Gaussian Naïve Bayes ===
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)

# Predictions
y_val_pred = nb_clf.predict(X_val)
y_test_pred = nb_clf.predict(X_test)

# Metrics
val_accuracy = accuracy_score(y_val, y_val_pred)
val_balanced_accuracy = balanced_accuracy_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print("\nNaive Bayes Validation Accuracy (Improved):", val_accuracy)
print("Naive Bayes Validation Balanced Accuracy (Improved):", val_balanced_accuracy)
print("Naive Bayes Validation F1-Score (Improved):", val_f1)

print("\nNaive Bayes Test Accuracy (Improved):", test_accuracy)
print("Naive Bayes Test Balanced Accuracy (Improved):", test_balanced_accuracy)
print("Naive Bayes Test F1-Score (Improved):", test_f1)
