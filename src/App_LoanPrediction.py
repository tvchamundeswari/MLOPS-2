import Inference

if __name__ == "__main__":
    prediction = Inference.predict_loan_servicability(
        "0", "0", "1000", "50000"
    )
    print("Prediction on Loan Servicability: ", prediction)
