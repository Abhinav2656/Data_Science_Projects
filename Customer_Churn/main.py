import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data = data.dropna()
data = data.reset_index(drop=True)

data_encoded = pd.get_dummies(data, drop_first=True)

X = data_encoded.drop(columns=['Churn_Yes'])
y = data_encoded['Churn_Yes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

coefficients = pd.DataFrame({'Feature': X.columns, 'Importance': model.coef_[0]})
coefficients = coefficients.sort_values(by='Importance', ascending=False)

print("\nTop 10 features driving churn:\n", coefficients.head(10))

def visualize_churn_distribution():
    sns.countplot(x='Churn', data=data, palette='viridis')
    plt.title('Churn Distribution')
    plt.xlabel('Churn (No = 0, Yes= 1)')
    plt.ylabel('Count')
    plt.show()

def visualize_contract_churn():
    sns.countplot(x='Contract', hue='Churn', data=data, palette='muted')
    plt.title('Churn by Contract Type')
    plt.show()

def visualize_monthly_charges():
    sns.boxplot(x='Churn', y='MonthlyCharges', data=data, palette='coolwarm')
    plt.title('Monthly Charges VS Churn')
    plt.show()

def visualize_tenure():
    sns.histplot(data=data, x='tenure', hue='Churn', kde=True, palette='coolwarm', bins=20)
    plt.title('Tenure Distribution by Churn')
    plt.xlabel('Tenure (Months)')
    plt.show()

def visualize_feature_importance():
    sns.barplot(x='Importance', y='Feature', data=coefficients.head(10), palette='coolwarm')
    plt.title('Top Features Driving Churn')
    plt.show()

visualize_churn_distribution()
visualize_contract_churn()
visualize_tenure()
visualize_feature_importance()
visualize_monthly_charges()