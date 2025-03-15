import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# === Load the feature-engineered dataset ===
df = pd.read_csv('Feature_Engineered_Data.csv')

print("Dataset shape:", df.shape)
print("Dataset columns:", df.columns.tolist())

# === Data Cleaning (minimal as data is already preprocessed) ===
# Fill any remaining missing values if they exist
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object', 'bool']).columns

for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

# === Prepare the Data for Modeling ===
# Separate target variable
y = df['Loan_Status']
X = df.drop('Loan_Status', axis=1)

# Convert boolean columns to int (if any)
bool_cols = X.select_dtypes(include=['bool']).columns
for col in bool_cols:
    X[col] = X[col].astype(int)

# Encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Train/Test Split (60% Train, 20% Val, 20% Test)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

print("\nTraining set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)
print("Test set shape:", X_test.shape)

# === Random Forest Model Training (no hyperparameter tuning) ===
# Using same parameters as baseline
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# === Predictions and Evaluation ===
y_val_pred = rf_clf.predict(X_val)
y_test_pred = rf_clf.predict(X_test)

# Validation metrics
val_accuracy = accuracy_score(y_val, y_val_pred)
val_balanced_accuracy = balanced_accuracy_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred)

# Test metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print("\n=== Model Performance ===")
print("\nValidation Metrics:")
print("Random Forest Validation Accuracy:", val_accuracy)
print("Random Forest Validation Balanced Accuracy:", val_balanced_accuracy)
print("Random Forest Validation F1-Score:", val_f1)

print("\nTest Metrics:")
print("Random Forest Test Accuracy:", test_accuracy)
print("Random Forest Test Balanced Accuracy:", test_balanced_accuracy)
print("Random Forest Test F1-Score:", test_f1)

# === Feature Importance Analysis (Simplified) ===
# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_clf.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Get top 15 features
top_15_features = feature_importance.head(15)

# Print only once
print("\nTop 15 Feature Importance:")
print(top_15_features)

# Visualize top 15 features
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=top_15_features)
plt.title('Top 15 Features by Importance')
plt.tight_layout()
plt.savefig('top_15_features.png')
plt.close()

print("Top 15 features visualization saved to 'top_15_features.png'")