#om 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import Inference

if __name__ == "__main__":
    prediction = Inference.predict_loan_servicability("0","0","1000","50000")
    print("Prdiction on Load Servicability: ", prediction)
