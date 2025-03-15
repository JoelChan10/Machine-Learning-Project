import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# === Load the dataset ===
df = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')

# === Data Cleaning ===
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# === Preprocessing: Normalization ===
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# === Prepare the Data for Modeling ===
X = df[numeric_cols]  # Only numeric features
y = df['Loan_Status']

le = LabelEncoder()
y = le.fit_transform(y)

# Train/Test Split (60% Train, 20% Val, 20% Test)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# === Random Forest Model Training ===
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Predictions
y_val_pred = rf_clf.predict(X_val)
y_test_pred = rf_clf.predict(X_test)

# Metrics
val_accuracy = accuracy_score(y_val, y_val_pred)
val_balanced_accuracy = balanced_accuracy_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print("\nRandom Forest Validation Accuracy:", val_accuracy)
print("Random Forest Validation Balanced Accuracy:", val_balanced_accuracy)
print("Random Forest Validation F1-Score:", val_f1)

print("\nRandom Forest Test Accuracy:", test_accuracy)
print("Random Forest Test Balanced Accuracy:", test_balanced_accuracy)
print("Random Forest Test F1-Score:", test_f1)