import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# === Load the feature-engineered dataset ===
print("Loading data...")
df = pd.read_csv('Feature_Engineered_Data.csv')

print("Dataset shape:", df.shape)

# === Data preparation ===
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

# === STEP 1: Train initial model to identify important features ===
print("\nTraining initial model to identify important features...")
initial_rf = RandomForestClassifier(n_estimators=100, random_state=42)
initial_rf.fit(X_train, y_train)

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': initial_rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop 15 important features:")
print(feature_importance.head(15))

# Manually select most important features
selected_features = [
    'Credit_History',
    'Good_Credit_History',
    'IncomeLoanRatio',
    'Log_TotalIncome',
    'ApplicantIncome',
    'Log_LoanAmount',
    'CoapplicantIncome'
]

# Create datasets with selected features
X_train_selected = X_train[selected_features]
X_val_selected = X_val[selected_features]
X_test_selected = X_test[selected_features]

# === STEP 2: Hyperparameter Tuning with GridSearchCV ===
print("\nPerforming hyperparameter tuning...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Perform grid search
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,  # 3-fold cross-validation
    scoring='f1',
    n_jobs=-1,  # Use all available cores
    verbose=1
)

grid_search.fit(X_train_selected, y_train)

# Get best parameters
best_params = grid_search.best_params_
print("\nBest hyperparameters:", best_params)

# === STEP 3: Train final model with best parameters on top features ===
optimized_model = RandomForestClassifier(random_state=42, **best_params)
optimized_model.fit(X_train_selected, y_train)

# === STEP 4: Evaluate model ===
# Validation set performance
y_val_pred = optimized_model.predict(X_val_selected)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_balanced_accuracy = balanced_accuracy_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred)

# Test set performance
y_test_pred = optimized_model.predict(X_test_selected)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"\n=== Optimized Model Performance (Top Features + Tuned Hyperparameters) ===")
print(f"Random Forest Validation Accuracy: {val_accuracy:.4f}")
print(f"Random Forest Validation Balanced Accuracy: {val_balanced_accuracy:.4f}")
print(f"Random Forest Validation F1-Score: {val_f1:.4f}")
print(f"\nRandom Forest Test Accuracy: {test_accuracy:.4f}")
print(f"Random Forest Test Balanced Accuracy: {test_balanced_accuracy:.4f}")
print(f"Random Forest Test F1-Score: {test_f1:.4f}")

# === STEP 5: Create visualization for F1-scores ===
plt.figure(figsize=(10, 6))

# Create bar chart
metrics = ['Validation F1-Score', 'Test F1-Score']
values = [val_f1, test_f1]
colors = ['skyblue', 'lightcoral']

plt.bar(metrics, values, color=colors)

# Add labels and title
plt.ylabel('F1-Score', fontweight='bold', fontsize=12)
plt.title('Validation vs Test F1-Score for Optimized Model', fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on bars
for i, v in enumerate(values):
    plt.text(i, v+0.01, f'{v:.4f}', ha='center', fontsize=12, fontweight='bold')

plt.ylim(0, max(values) + 0.1)  # Add some space above the bars for labels
plt.tight_layout()
plt.savefig('optimized_model_f1_scores.png')
plt.show()

print("\nF1-score visualization saved as 'optimized_model_f1_scores.png'")