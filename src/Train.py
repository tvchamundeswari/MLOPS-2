from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
import joblib

import EDA
import Preprocessor

def train_and_evaluate():
    """
    Function to train an SVM model on preprocessed loan data and evaluate its performance.
    """

    # Load and preprocess the dataset
    df = Preprocessor.load_data()
    df = EDA.performEDA(df)

    # Drop the 'Loan_ID' column
    df.drop('Loan_ID', axis=1, inplace=True)

    # Prepare features and target variable
    features = df.drop('Loan_Status', axis=1)
    target = df['Loan_Status'].values

    # Split data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(
        features, target, test_size=0.2, random_state=10
    )

    # Handle class imbalance using oversampling
    ros = RandomOverSampler(sampling_strategy='minority', random_state=0)
    X_resampled, Y_resampled = ros.fit_resample(X_train, Y_train)

    print("Dataset: Training data sample:\n", X_train.head())

    # Normalize the feature set
    scaler = StandardScaler()
    X_resampled = scaler.fit_transform(X_resampled)
    X_val = scaler.transform(X_val)

    # Define hyperparameter tuning with GridSearchCV
    svc_parameters = {'kernel': ['linear', 'rbf'], 'C': [4, 5, 6, 7, 10, 15]}
    model_svc = SVC()

    clf = GridSearchCV(model_svc, svc_parameters, cv=10, scoring='accuracy')
    print("Starting GridSearchCV...")
    clf.fit(X_resampled, Y_resampled)

    # Best hyperparameters
    print("Best parameters found:", clf.best_params_)
    print("Best cross-validation score:", clf.best_score_)

    # Train final model with best hyperparameters
    final_model = SVC(kernel=clf.best_params_['kernel'], C=clf.best_params_['C'])
    final_model.fit(X_resampled, Y_resampled)

    # Evaluate model performance
    train_auc = metrics.roc_auc_score(Y_resampled, final_model.predict(X_resampled))
    val_auc = metrics.roc_auc_score(Y_val, final_model.predict(X_val))

    print(f'Training AUC Score: {train_auc:.4f}')
    print(f'Validation AUC Score: {val_auc:.4f}')

    # Save the trained model
    joblib.dump(final_model, 'model.joblib')
    print("Model saved as 'model.joblib'.")

    # Test prediction
    sample_input = [[0, 0, 1000, 50000]]  # Ensure numeric inputs
    prediction = final_model.predict(sample_input)
    print("Sample Prediction:", prediction)

    # Model accuracy
    accuracy = final_model.score(X_val, Y_val)
    print('Model Accuracy:', accuracy)

    # Confusion Matrix
    cm = metrics.confusion_matrix(Y_val, final_model.predict(X_val))
    print("Confusion Matrix:\n", cm)

    # Classification Report
    print("Classification Report:\n", metrics.classification_report(Y_val, final_model.predict(X_val)))


if __name__ == "__main__":
    train_and_evaluate()
