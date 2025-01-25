import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder

import warnings

# Ignore warnings
warnings.filterwarnings('ignore')


def perform_eda(df):
    """
    Perform Exploratory Data Analysis (EDA) on the given DataFrame.
    """
    df = pd.read_csv('src/loan_data.csv')

    # Basic dataset information
    print(df.head())
    print(df.shape)
    print(df.info())
    print(df.describe())

    # Loan Status Pie Chart
    temp = df['Loan_Status'].value_counts()
    plt.pie(
        temp.values, 
        labels=temp.index, 
        autopct='%1.1f%%'
    )
    plt.title("Loan Status Distribution")
    plt.show()

    # Count plots for categorical features
    plt.subplots(figsize=(15, 5))
    for i, col in enumerate(['Gender', 'Married']):
        plt.subplot(1, 2, i + 1)
        sb.countplot(data=df, x=col, hue='Loan_Status')
    plt.tight_layout()
    plt.show()

    # Distribution plots for numerical features
    plt.subplots(figsize=(15, 5))
    for i, col in enumerate(['ApplicantIncome', 'LoanAmount']):
        plt.subplot(1, 2, i + 1)
        sb.histplot(df[col], kde=True)
    plt.tight_layout()
    plt.show()

    # Box plots for numerical features
    plt.subplots(figsize=(15, 5))
    for i, col in enumerate(['ApplicantIncome', 'LoanAmount']):
        plt.subplot(1, 2, i + 1)
        sb.boxplot(x=df[col])
    plt.tight_layout()
    plt.show()

    # Filter data to remove outliers
    df = df[df['ApplicantIncome'] < 25000]
    df = df[df['LoanAmount'] < 400000]

    # Grouping and analyzing the data
    loan_by_gender = df.groupby('Gender').mean(numeric_only=True)['LoanAmount']
    loan_by_marital_status = df.groupby(['Married', 'Gender']).mean(numeric_only=True)['LoanAmount']

    # Function to apply label encoding
    def encode_labels(data):
        for col in data.columns:
            if data[col].dtype == 'object':
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
        return data

    # Apply label encoding to categorical columns
    df = encode_labels(df)

    # Generating Heatmap for correlation analysis
    sb.heatmap(df.corr() > 0.8, annot=True, cbar=False)
    plt.title("Feature Correlation Heatmap")
    plt.show()

    return df
