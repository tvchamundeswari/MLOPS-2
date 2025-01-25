from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
import joblib

import EDA as EDA
import Preprocessor as Preprocessor

#Loading the dataset and prprocessin
df = Preprocessor.load_data()
df = EDA.performEDA(df)
df.drop('Loan_ID',axis=1,inplace=True)

# Start Model Training
features = df.drop('Loan_Status', axis=1)
target = df['Loan_Status'].values

X_train, X_val,	Y_train, Y_val = train_test_split(features, target,
									test_size=0.2,
									random_state=10)

# As the data was highly imbalanced we will balance
# it by adding repetitive rows of minority class.
ros = RandomOverSampler(sampling_strategy='minority',
						random_state=0)
X, Y = ros.fit_resample(X_train, Y_train)

#X_train.shape, X.shape
print("Dataset : training data sample :", X_train.head())

#Normalizing the features for stable and fast training.
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_val = scaler.transform(X_val)

#Apply GridSearchCV for best model parameters:
from sklearn.model_selection import GridSearchCV

# Assigning the parameters and its values which need to be tuned.
svc_parameters = {'kernel': ['linear', 'rbf'], 'C':[4,5,6,7,10,15]}
 
# Fitting the SVM model
modelsvc = SVC()
 
# Performing the GridSearchCV
clf = GridSearchCV(modelsvc, svc_parameters, cv = 10, scoring='accuracy')
print("Starting the gridsearchcv")
clf.fit(X, Y) #, )

print(clf.best_params_)

print(clf.best_score_)

final_model = SVC(kernel=clf.best_params_['kernel'], C=clf.best_params_['C'])
final_model.fit(X, Y)

print('Training Accuracy : ', metrics.roc_auc_score(Y, final_model.predict(X)))
print('Validation Accuracy : ', metrics.roc_auc_score(Y_val, final_model.predict(X_val)))
print()

print("Save the model")
joblib.dump(final_model,'model.joblib')

## tEST
print("Prdiction : ", final_model.predict([["0","0","1000","50000"]]))

score = final_model.score(X_val, Y_val)
print('Accuracy: ', score)

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
#training_roc_auc = roc_auc_score(Y, model.predict(X))
#validation_roc_auc = roc_auc_score(Y_val, model.predict(X_val))
#print('Training ROC AUC Score:', training_roc_auc)
#print('Validation ROC AUC Score:', validation_roc_auc)
#print()
cm = confusion_matrix(Y_val, final_model.predict(X_val))

from sklearn.metrics import classification_report
print(classification_report(Y_val, final_model.predict(X_val)))

#