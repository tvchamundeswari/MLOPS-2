import joblib


def predict_loan_servicability(gender, marital_status, income, loan_amount):
    model = joblib.load('model.joblib')
    prediction = model.predict([[gender, marital_status, income, loan_amount]])
    print(
        "Loan servicability for:",
        gender, marital_status, income, loan_amount,
        "is:", prediction
    )
    return prediction
