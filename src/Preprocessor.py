# prompt: delete specific column in a csv
import pandas as pd

def load_data():
    # Load the CSV file into a Pandas DataFrame
    df = pd.read_csv("src/loan_data.csv")
    # Drop the specified column
    #df = df.drop('Dependents', axis=1)
    #df = df.drop('Education', axis=1)
    #df = df.drop('Self_Employed', axis=1)
    #df = df.drop('CoapplicantIncome', axis=1)
    #df = df.drop('Loan_Amount_Term', axis=1)
    #df = df.drop('Credit_History', axis=1)
    #df = df.drop('Property_Area', axis=1)
    
    # Save the modified DataFrame to a new CSV file
    #df.to_csv('loan_data.csv', index=False)
    return df