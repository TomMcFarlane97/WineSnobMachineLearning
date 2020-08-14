import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from os import path

# Following this tutorial - https://elitedatascience.com/python-machine-learning-tutorial-scikit-learn

# Step 3: Load red wine data and read.
from sklearn.pipeline import make_pipeline

datasetCSV = '/Users/tom.mcfarlane/Developer/Personal/WineSnobMachineLearning/datasets/red-wine-quality.csv'
if not path.isfile(datasetCSV):
    print(datasetCSV + ' is not a correct path')
    exit(1)

data = pd.read_csv(datasetCSV)

# Step 4: Split data into training and test sets.
targetFeatures = data.quality
inputFeatures = data.drop('quality', axis=1)
X_train, X_test, y_train, y_test = \
    train_test_split(inputFeatures, targetFeatures, test_size=0.2, random_state=123, stratify=targetFeatures)

# Step 5: Declare data preprocessing steps.
scaler = preprocessing.StandardScaler().fit(X_train)
X_test_scaled = scaler.transform(X_test)
# print X_test_scaled.mean(axis=0)

# print X_test_scaled.std(axis=0)

pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))

# Step 6: Declare hyperparameters to tune.
hyperParameters = {
    'randomforestregressor__max_features': ['auto', 'sqrt', 'log2'],
    'randomforestregressor__max_depth': [None, 5, 3, 1]
}

clf = GridSearchCV(pipeline, hyperParameters, cv=10)
# Fit and tune model
clf.fit(X_train, y_train)
print clf.best_params_

# Step 8: Refit on the entire training set.
print clf.refit

# predict new test set
y_pred = clf.predict(X_test)

print r2_score(y_test, y_pred)
print mean_squared_error(y_test, y_pred)

# Step 10: Save model for future use.

joblib.dump(clf, 'rf_regressor.pkl')
clf2 = joblib.load('rf_regressor.pkl')

# Predict data set using loaded model
clf2.predict(X_test)


