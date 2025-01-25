from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
import joblib
import EDA as EDA
import Preprocessor as Preprocessor

# Loading the dataset and preprocessing
df = Preprocessor.load_data()
df = EDA.performEDA(df)
df.drop('Loan_ID', axis=1, inplace=True)

# Start Model Training
features = df.drop('Loan_Status', axis=1)
target = df['Loan_Status'].values

X_train, X_val, Y_train, Y_val = train_test_split(
    features, target, test_size=0.2, random_state=10
)

# Balancing the data by oversampling
ros = RandomOverSampler(sampling_strategy='minority', random_state=0)
X, Y = ros.fit_resample(X_train, Y_train)

print("Dataset: training data sample:", X_train.head())

# Normalizing the features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_val = scaler.transform(X_val)

# Apply GridSearchCV for best model parameters
svc_parameters = {'kernel': ['linear', 'rbf'], 'C': [4, 5, 6, 7, 10, 15]}
modelsvc = SVC()
clf = GridSearchCV(modelsvc, svc_parameters, cv=10, scoring='accuracy')

print("Starting the grid search CV")
clf.fit(X, Y)

print(clf.best_params_)
print(clf.best_score_)

final_model = SVC(kernel=clf.best_params_['kernel'], C=clf.best_params_['C'])
final_model.fit(X, Y)

print('Training Accuracy:', metrics.roc_auc_score(Y, final_model.predict(X)))
print('Validation Accuracy:', metrics.roc_auc_score(Y_val, final_model.predict(X_val)))

print("Save the model")
joblib.dump(final_model, 'model.joblib')

print("Prediction:", final_model.predict([[0, 0, 1000, 50000]]))

score = final_model.score(X_val, Y_val)
print('Accuracy:', score)

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(Y_val, final_model.predict(X_val))
print(classification_report(Y_val, final_model.predict(X_val)))
