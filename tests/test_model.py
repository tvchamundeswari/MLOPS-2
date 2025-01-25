import unittest
import joblib


class TestModel(unittest.TestCase):

    def test_model_presence(self):
        model = joblib.load('model.joblib')
        self.assertFalse(model is None)

    def test_model_prediction(self):
        self.assertEqual('foo'.upper(), 'FOO')


if __name__ == '__main__':
    unittest.main()
