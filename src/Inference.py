import joblib

def predict_loan_servicability(gender, marrital_Status, income, loan_amount):
    model = joblib.load('model.joblib')
    prediction = model.predict([[gender, marrital_Status, income, loan_amount]])
    print("Loan servicability for :", gender, marrital_Status, income, loan_amount, " is : " ,prediction)
    return prediction