import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
fe_df = pd.read_csv("Feature_Engineered_Data.csv")

df_eligible_loan = df[df.Loan_Status == "Y"]
df_no_loan = df[df.Loan_Status == "N"]

df_married = df[df.Married == 'Yes']

df_graduates = df[df.Education == 'Graduate']
df_not_graduates = df[df.Education != 'Graduate']

# Generate the size of each "family"

# Assume 3+ dependents as a max of 4 dependents
df_family = df.copy()
df_family['Dependents'] = df_family['Dependents'].replace('3+', '4').astype(float)

# Create 'family_size' based on 'Dependents' and 'Married'
df_family['family_size'] = 1 + df_family['Dependents']  # Start with applicant themselves
df_family.loc[df_family['Married'] == 'Yes', 'family_size'] = df_family.loc[df_family['Married'] == 'Yes', 'family_size'] + 1

"""## Summary of Income, Partner Income & Loan Amount of Loanees"""

def calculate_stats(series):
  """Calculates mean, median, and mode for a Pandas Series."""
  mean = series.mean()
  median = series.median()
  mode = series.mode()[0]
  return mean, median, mode

# Calculate for ApplicantIncome
mean_applicant, median_applicant, mode_applicant = calculate_stats(df_eligible_loan['ApplicantIncome'])
print("Applicant Income:")
print(f"Mean:\t", format(mean_applicant, ".2f"))
print(f"Median:\t", format(median_applicant, ".2f"))
print(f"Mode:\t", format(mode_applicant, ".2f"))

# Calculate for CoapplicantIncome
mean_coapplicant, median_coapplicant, mode_coapplicant = calculate_stats(df_married['CoapplicantIncome'])
print("\nCoapplicant Income:")
print(f"Mean:\t", format(mean_coapplicant, ".2f"))
print(f"Median:\t", format(median_coapplicant, ".2f"))
print(f"Mode:\t", format(mode_coapplicant, ".2f"))


# Calculate for LoanAmount
mean_loan, median_loan, mode_loan = calculate_stats(df_eligible_loan['LoanAmount'])
print("\nLoan Amount:")
print(f"Mean:\t", format(mean_loan, ".2f"))
print(f"Median:\t", format(median_loan, ".2f"))
print(f"Mode:\t", format(mode_loan, ".2f"))

"""## Visualise Applicant Income to Loan Amount"""

# Create a scatter plot to visualize the relationship
plt.figure(figsize=(8, 4))  # Adjust the figure size as needed
sns.scatterplot(x='ApplicantIncome', y='LoanAmount', data=df)

# Add labels and title
plt.xlabel('Applicant Income')
plt.ylabel('Loan Amount')
plt.title('Relationship Between Applicant Income and Loan Amount')

# Display the plot
plt.show()

"""## Credit Visualise Credit History of Loanees"""

cred_hist_no_loan_counts = df_no_loan.Credit_History.value_counts()
cred_hist_loan_counts = df_eligible_loan.Credit_History.value_counts()

fig, axes = plt.subplots(1, 2, figsize=(8, 6))

# Create a pie chart
axes[0].pie(cred_hist_no_loan_counts, labels=['Good CS', 'Bad CS'], autopct='%1.1f%%', startangle=90)
axes[0].title.set_text('Credit Score History\n(NOT Eligible for Loan)')

axes[1].pie(cred_hist_loan_counts, labels=['Good CS', 'Bad CS'], autopct='%1.1f%%', startangle=90)
axes[1].title.set_text('Credit Score History\n(Loan-Eligible)')

"""## Visualise Property Areas of Loanees"""

# Visualise Property Areas of Loanees using subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 2 columns

df_eligible_loan['Property_Area'].value_counts().plot(kind='bar', ax=axes[0], title="Eligible for Loan")
df_no_loan['Property_Area'].value_counts().plot(kind='bar', ax=axes[1], title="Not Eligible for Loan")

plt.tight_layout()
plt.show()

"""## Visualise Education Status of Loanees"""

# Count the occurrences of each education level
# Eligible for Loan
education_counts_eligible = df_eligible_loan['Education'].value_counts()
# Not Eligible for Loan
education_counts_no_loan = df_no_loan['Education'].value_counts()

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(8, 4))  # 1 row, 2 columns

# Pie chart for Eligible for Loan
axes[0].pie(education_counts_eligible, labels=education_counts_eligible.index, autopct='%1.1f%%', startangle=90)
axes[0].title.set_text('% of Education Levels (Eligible for Loan)')
axes[0].axis('equal')

# Pie chart for Not Eligible for Loan
axes[1].pie(education_counts_no_loan, labels=education_counts_no_loan.index, autopct='%1.1f%%', startangle=90)
axes[1].title.set_text('% of Education Levels (NOT Eligible for Loan)')
axes[1].axis('equal')

# Adjust layout and display
plt.tight_layout()  # Adjusts subplot params for a tight layout
plt.show()

"""## Visualise Loan Info of Graduates"""

# --- Get the range for LoanAmount for both graduates and non-graduates ---
loan_amount_min = min(df_graduates['LoanAmount'].min(), df_not_graduates['LoanAmount'].min())
loan_amount_max = max(df_graduates['LoanAmount'].max(), df_not_graduates['LoanAmount'].max())

# --- Get the range for Frequency for both graduates and non-graduates ---
freq_min = 0  # Assuming frequency starts from 0
# Calculate the maximum frequency of LoanAmount for graduates and non-graduates using value_counts().max()
freq_max_grads = df_graduates['LoanAmount'].value_counts(bins=20).max()
freq_max_non_grads = df_not_graduates['LoanAmount'].value_counts(bins=20).max()

freq_max = max(freq_max_grads, freq_max_non_grads)

# --- Visualizations for Graduates ---
# Count the occurrences of each loan status
loan_status_counts = df_graduates['Loan_Status'].value_counts()
fig, axes = plt.subplots(2, 2, figsize=(8, 6))  # 1 row, 2 columns

# Create a pie chart
axes[0][0].pie(loan_status_counts, labels=loan_status_counts.index, autopct='%1.1f%%', startangle=90)
axes[0][0].title.set_text('Loan Status % of Graduates')
axes[0][0].axis('equal')

# Loan Amount Distribution (with adjusted x-axis and y-axis limits)
axes[0][1].hist(df_graduates['LoanAmount'], bins=20, edgecolor='black')
axes[0][1].title.set_text('Distribution of Loan Amount for Graduates')
axes[0][1].set_xlabel('Loan Amount ($)')
axes[0][1].set_ylabel('Number of Loans')
axes[0][1].set_xlim(loan_amount_min, loan_amount_max)  # Set x-axis limits
axes[0][1].set_ylim(freq_min, freq_max) # Set y-axis limits

# --- Visualizations for Non-Graduates ---
loan_status_counts = df_not_graduates['Loan_Status'].value_counts()
# Create a pie chart
axes[1][0].pie(loan_status_counts, labels=loan_status_counts.index, autopct='%1.1f%%', startangle=90)
axes[1][0].title.set_text('Loan Status % of Non-Grads') # This title needs correction
axes[1][0].axis('equal')

# Loan Amount Distribution (with adjusted x-axis and y-axis limits)
axes[1][1].hist(df_not_graduates['LoanAmount'], bins=20, edgecolor='black')
axes[1][1].title.set_text('Distribution of Loan Amount for Non-Grads')
axes[1][1].set_xlabel('Loan Amount ($)')
axes[1][1].set_ylabel('Number of Loans')
axes[1][1].set_xlim(loan_amount_min, loan_amount_max)  # Set x-axis limits
axes[1][1].set_ylim(freq_min, freq_max) # Set y-axis limits


plt.tight_layout()
plt.show()

"""## Visualise Loan Amount across Family Size"""

plt.figure(figsize=(8, 5))

# Plot LoanAmount vs. family_size
sns.boxplot(x='family_size', y='LoanAmount', data=df_family)
plt.title('Loan Amount across Families')  # Set the title of the plot
plt.ylabel('Loan Amount ($)')  # Set the label for the y-axis
plt.xlabel('Family Size')

# Get the current x-axis tick labels
xticklabels = plt.gca().get_xticklabels()

# Modify the label for 6.0 to ">6.0"
for label in xticklabels:
    if label.get_text() == '6.0':
        label.set_text('>=6.0')

# Set the modified tick labels
plt.gca().set_xticklabels(xticklabels)

plt.show()

"""## Prep Data

### Prep Data for Model Training
"""

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['category', 'object']).columns

# Initialize LabelEncoder
le = LabelEncoder()

# Encode categorical columns
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

df['LoanAmount'] = df['LoanAmount'].fillna(0)
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(0)
df['Credit_History'] = df['Credit_History'].fillna(0)

"""### Choose 7 Feature Extracted Data"""

fe_df.drop('Loan_Amount_Term', axis=1, inplace=True)
fe_df.drop('TotalIncome', axis=1, inplace=True)
fe_df.drop('Loan_Amount_Term_Years', axis=1, inplace=True)
fe_df.drop('Gender_Male', axis=1, inplace=True)
fe_df.drop('Married_Yes', axis=1, inplace=True)
fe_df.drop('Education_Not Graduate', axis=1, inplace=True)
fe_df.drop('Self_Employed_Yes', axis=1, inplace=True)
fe_df.drop('Property_Area_Semiurban', axis=1, inplace=True)
fe_df.drop('Property_Area_Urban', axis=1, inplace=True)
fe_df.drop('Dependents', axis=1, inplace=True)
fe_df.drop('LoanAmount', axis=1, inplace=True)

"""### Split Data"""

# Assuming 'fe_df' is your DataFrame
# Separate features (X) and target variable (y)
base_X = df.drop('Loan_Status', axis=1)  # Features are all columns except 'Loan_Status'
base_y = df['Loan_Status']  # Target variable is 'Loan_Status'

# Split data into training and testing sets
base_X_train, base_X_test, base_y_train, base_y_test = train_test_split(base_X, base_y, test_size=0.2, random_state=42) # 80% train, 20% test

# Assuming 'fe_df' is your DataFrame
# Separate features (X) and target variable (y)
fe_X = fe_df.drop('Loan_Status', axis=1)  # Features are all columns except 'Loan_Status'
fe_y = fe_df['Loan_Status']  # Target variable is 'Loan_Status'

# Split data into training and testing sets
fe_X_train, fe_X_test, fe_y_train, fe_y_test = train_test_split(fe_X, fe_y, test_size=0.2, random_state=42) # 80% train, 20% test

"""## Support Vector Machines Model

### Train Baseline SVM Model
"""

# Initialize and train an SVM model
base_svm_model = SVC(kernel='rbf')
base_svm_model.fit(base_X_train, base_y_train)

# Make predictions on the test set
base_y_pred = base_svm_model.predict(base_X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(base_y_test, base_y_pred)
balanced_accuracy = balanced_accuracy_score(base_y_test, base_y_pred)
f1 = f1_score(base_y_test, base_y_pred, average='weighted')
print(f"Baseline SVM Model Accuracy: {accuracy}")
print(f"Baseline SVM Model Balanced Accuracy: {balanced_accuracy}")
print(f"Baseline SVM Model f1-score: {f1}")

"""### Train SVM Model using Feature Engineered Data"""

# Initialize and train an SVM model
fe_svm_model = SVC(kernel='rbf')
fe_svm_model.fit(fe_X_train, fe_y_train)

# Make predictions on the test set
fe_y_pred = fe_svm_model.predict(fe_X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(fe_y_test, fe_y_pred)
balanced_accuracy = balanced_accuracy_score(fe_y_test, fe_y_pred)
f1 = f1_score(fe_y_test, fe_y_pred, average='weighted')
print(f"Improved SVM Model Accuracy: {accuracy}")
print(f"Improved SVM Model Balanced Accuracy: {balanced_accuracy}")
print(f"Improved SVM Model f1-score: {f1}")

"""### Train SVM Model using 7 Feature Extracted Data"""

# Initialize and train an SVM model
fe_svm_model = SVC(kernel='rbf')
fe_svm_model.fit(fe_X_train, fe_y_train)

# Make predictions on the test set
fe_y_pred = fe_svm_model.predict(fe_X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(fe_y_test, fe_y_pred)
balanced_accuracy = balanced_accuracy_score(fe_y_test, fe_y_pred)
f1 = f1_score(fe_y_test, fe_y_pred, average='weighted')
print(f"Improved SVM Model Accuracy: {accuracy}")
print(f"Improved SVM Model Balanced Accuracy: {balanced_accuracy}")
print(f"Improved SVM Model f1-score: {f1}")

"""### Hyperparameter Tune Better SVM Model"""

# Define the parameter grid to search
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'gamma': [1, 0.1, 0.01, 0.001], # Kernel coefficient
    'kernel': ['rbf', 'poly'] # Kernel type
}

# Create a GridSearchCV object
grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

# Fit the grid search to the training data
grid_search.fit(fe_X_train, fe_y_train)

# Evaluate the best model on the test data
best_fe_svm_model = grid_search.best_estimator_
fe_y_pred = best_fe_svm_model.predict(fe_X_test)
accuracy = accuracy_score(fe_y_test, fe_y_pred)
balanced_accuracy = balanced_accuracy_score(fe_y_test, fe_y_pred)
f1 = f1_score(fe_y_test, fe_y_pred, average='weighted')

"""### Model Evaluation"""

# Print the best parameters and the best score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_}")
print(f"Best Feature Engineered SVM Model Accuracy: {accuracy}\n")

print(f"Tuned & Improved SVM Model Accuracy: {accuracy}")
print(f"Tuned & Improved SVM Model Balanced Accuracy: {balanced_accuracy}")
print(f"Tuned & Improved SVM Model f1-score: {f1}")

y_pred = best_fe_svm_model.predict(fe_X_test)
cm = confusion_matrix(fe_y_test, y_pred)

sns.heatmap(
    cm,
    cmap="Blues",
    annot=True,
    fmt='d',
    xticklabels=['Negative', 'Positive'],
    yticklabels=['Negative', 'Positive']
    )
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Access data from grid search results
mean_test_scores = grid_search.cv_results_['mean_test_score']
params = grid_search.cv_results_['params']

# Group data by kernel
rbf_scores = []
rbf_labels = []
poly_scores = []
poly_labels = []

for i in range(len(params)):
    if params[i]['kernel'] == 'rbf':
        rbf_scores.append(mean_test_scores[i])
        rbf_labels.append(str(params[i]))
    else:  # Assuming 'poly' is the other kernel
        poly_scores.append(mean_test_scores[i])
        poly_labels.append(str(params[i]))

# --- Find the overall minimum and maximum mean test scores ---
min_score = min(min(rbf_scores), min(poly_scores))
max_score = max(max(rbf_scores), max(poly_scores))

fig, axes = plt.subplots(1, 2, figsize=(8, 6))  # 1 row, 2 columns

# --- Plot for 'rbf' kernel ---
axes[0].plot(rbf_scores, label='rbf', marker='o')
axes[0].set_xlabel('Parameter Combination')
axes[0].set_ylabel('Mean Test Score')
axes[0].set_title('Mean Test Score vs. Parameter Combination\n(rbf Kernel)')
axes[0].set_xticks(range(len(rbf_labels)))
axes[0].set_xticklabels(rbf_labels, rotation=90)
axes[0].set_ylim(min_score, max_score)  # Set y-axis limits

# --- Plot for 'poly' kernel ---
axes[1].plot(poly_scores, label='poly', marker='x')
axes[1].set_xlabel('Parameter Combination')
axes[1].set_ylabel('Mean Test Score')
axes[1].set_title('Mean Test Score vs. Parameter Combination\n(poly Kernel)')
axes[1].set_xticks(range(len(poly_labels)))
axes[1].set_xticklabels(poly_labels, rotation=90)
axes[1].set_ylim(min_score, max_score)  # Set y-axis limits

# Adjust layout and display
plt.tight_layout()
plt.show()
