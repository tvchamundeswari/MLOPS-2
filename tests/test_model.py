import unittest
import joblib
import os
import numpy as np

class TestLoanPredictionModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """ Load the trained model once for all tests. """
        model_path = 'model.joblib'
        if os.path.exists(model_path):
            cls.model = joblib.load(model_path)
        else:
            cls.model = None

    def test_model_presence(self):
        """ Test if the model file exists and can be loaded. """
        self.assertIsNotNone(self.model, "Model file 'model.joblib' is missing or failed to load.")

    def test_model_prediction(self):
        """ Test if the model produces predictions correctly for a sample input. """
        sample_input = np.array([[0, 0, 1000, 50000]])  # Ensure numeric input format
        prediction = self.model.predict(sample_input)
        self.assertIn(prediction[0], [0, 1], "Prediction should be either 0 or 1.")

    def test_model_type(self):
        """ Ensure the loaded object is a valid model. """
        from sklearn.base import BaseEstimator
        self.assertIsInstance(self.model, BaseEstimator, "Loaded model is not a valid scikit-learn model.")

if __name__ == '__main__':
    unittest.main()
