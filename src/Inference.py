import joblib


def predict_loan_servicability(gender, marital_status, income, loan_amount):
    """
    Predict loan serviceability based on input features.

    Args:
        gender (int): 0 for Female, 1 for Male.
        marital_status (int): 0 for Single, 1 for Married.
        income (float): Applicant's income.
        loan_amount (float): Requested loan amount.

    Returns:
        int: Predicted loan serviceability (0 or 1).
    """

    # Load trained model
    model_path = 'src/model.joblib'
    
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        return None

    # Prepare input data
    input_features = [[gender, marital_status, income, loan_amount]]
    prediction = model.predict(input_features)

    print(f"Loan servicability for Gender:{gender}, Marital Status:{marital_status}, "
          f"Income:{income}, Loan Amount:{loan_amount} => Prediction: {prediction[0]}")

    return prediction[0]
