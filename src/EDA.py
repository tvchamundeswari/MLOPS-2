import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder

import warnings

warnings.filterwarnings('ignore')


def performEDA(df):
    df = pd.read_csv('src/loan_data.csv')
    df.head()
    df.shape
    df.info()
    df.describe()

    temp = df['Loan_Status'].value_counts()
    plt.pie(temp.values, labels=temp.index, autopct='%1.1f%%')
    plt.show()

    plt.subplots(figsize=(15, 5))
    for i, col in enumerate(['Gender', 'Married']):
        plt.subplot(1, 2, i + 1)
        sb.countplot(data=df, x=col, hue='Loan_Status')
    plt.tight_layout()
    plt.show()

    plt.subplots(figsize=(15, 5))
    for i, col in enumerate(['ApplicantIncome', 'LoanAmount']):
        plt.subplot(1, 2, i + 1)
        sb.distplot(df[col])
    plt.tight_layout()
    plt.show()

    plt.subplots(figsize=(15, 5))
    for i, col in enumerate(['ApplicantIncome', 'LoanAmount']):
        plt.subplot(1, 2, i + 1)
        sb.boxplot(df[col])
    plt.tight_layout()
    plt.show()

    df = df[df['ApplicantIncome'] < 25000]
    df = df[df['LoanAmount'] < 400000]

    print(df.groupby('Gender').mean(numeric_only=True)['LoanAmount'])
    print(df.groupby(['Married', 'Gender']).mean(numeric_only=True)['LoanAmount'])

    # Function to apply label encoding
    def encode_labels(data):
        for col in data.columns:
            if data[col].dtype == 'object':
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])

        return data

    # Applying function in whole column
    df = encode_labels(df)

    # Generating Heatmap
    sb.heatmap(
        df.corr() > 0.8,
        annot=True,
        cbar=False
    )
    plt.show()

    return df
