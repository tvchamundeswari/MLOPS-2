import unittest
import joblib


class TestModel(unittest.TestCase):

    def test_model_presence(self):
        model = joblib.load('model.joblib')
        self.assertFalse(model is None)

    def test_model_prediction(self):
        model = joblib.load('model.joblib')
        prediction = model.predict([["0", "0", "1000", "50000"]])
        self.assertTrue(prediction ,[1])


if __name__ == '__main__':
    unittest.main()
