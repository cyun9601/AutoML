# %% Import Module 
import numpy as np
import pandas as pd 
from sklearn.linear_model import LogisticRegression

# %% Data Load and Preprocessing

data = pd.read_csv("./data_files/diabetes.csv")

data[["Glucose", "BloodPressure", "SkinThickness",	"Insulin", "BMI"]] = data[["Glucose", "BloodPressure", "SkinThickness",	"Insulin", "BMI"]].replace(0, np.NaN)

# Fill missing values with mean column values
data.fillna(data.mean(), inplace=True)

# Count the number of NaN values in each column
print(data.isnull().sum())

# Split dataset into inputs and outputs 
values = data.values
X = values[:, 0:8]
Y = values[:, 8]

# %% Data Load and Preprocessing

# initiate the LR model with random hyperparameters 
lr = LogisticRegression(penalty='l2', dual=False, max_iter=110)
lr.fit(X, Y)

print("lr score : ", lr.score(X, Y))

# %% KFold

# You will need the following dependencies for applying Cross-Validation and evaluating the cross-validated score
from sklearn.model_selection import KFold, cross_val_score

# Build the k-fold cross-validator
kfold = KFold(n_splits=3, random_state=7, shuffle=True)

result = cross_val_score(lr, X, Y, cv=kfold, scoring='accuracy')
print(result.mean())


# %% Gridsearch 
from sklearn.model_selection import GridSearchCV
import time 

dual = [True, False]
max_iter = [100, 110, 120, 130, 140]
C = [1.0, 1.5, 2.0, 2.5]
param_grid = dict(dual=dual, max_iter = max_iter, C=C)

lr = LogisticRegression(penalty='l2')
grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv=3, n_jobs=-1)

start_time = time.time()
grid_result = grid.fit(X, Y)

# Summarize results 
print("Best: %f using %s" %(grid_result.best_score_, grid_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')

# %% Random Search 
from sklearn.model_selection import RandomizedSearchCV

random = RandomizedSearchCV(estimator=lr, param_distributions=param_grid, cv=3, n_jobs=-1)

start_time = time.time()
random_result = random.fit(X, Y)

# Summarize results 
print("Best: %f using %s" %(random_result.best_score_, random_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')
