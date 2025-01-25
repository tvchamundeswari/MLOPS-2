#om
#from src.model import train_model
import unittest
import joblib

class TestStringMethods(unittest.TestCase):

    def test_model_presence(self):
        model = joblib.load('model.joblib')
        self.assertFalse(model is None)

    def test_model_rediction(self):
        self.assertEqual('foo'.upper(), 'FOO')

if __name__ == '__main__':
    unittest.main()
