# SVM Model Files

import pandas as pd # data storage
import numpy as np  # math and stuff

import sklearn  
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

from sklearn.metrics import median_absolute_error, max_error, mean_squared_error

from sklearn import svm
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

def svm_gz(X_train, X_test, y_train, y_test, jobs=4, OS='os'):
    # This is a SVM, let's scale this data!
    n_feat = np.shape(X_train)[1]
  
    scaler_X = preprocessing.StandardScaler().fit(X_train, n_feat)
    scaler_y = preprocessing.StandardScaler().fit(y_train)

    X_scaled_train = scaler_X.transform(X_train)
    X_scaled_test  = scaler_X.transform(X_test) # use the same transformer
    Y_scaled_train = scaler_y.transform(y_train)

    # SVM Model
    svm_rbf = svm.SVR(kernel='rbf')
    svm_rbf.fit(X_scaled_train, Y_scaled_train)
    preds = svm_rbf.predict(X_scaled_test)

    preds_transformed = scaler_y.inverse_transform(preds.reshape(-1, 1))

    rmse1 = mean_squared_error(y_test, preds_transformed, squared=False)
    print("Mean Squared Error: %f" % (rmse1))
    max1 = max_error(y_test, preds_transformed)
    print("Max Error: %f" % (max1))
    MAE1 = median_absolute_error(y_test, preds_transformed)
    print("Median Abs Error: %f" % (MAE1))

    parameters = {
        'C': [ 2, 3, 4, 5],
        'tol':[0.0005,0.001, .1, 0.05, 0.2],
        'gamma': [ 'auto', 'scale', 0.5, 1.5, 3],
        'max_iter': [200, 400, 600],
        'epsilon': [ 0.2, 0.4, 0.8],
        'degree': [2, 3, 4]}

    estimator = svm.SVR(kernel='rbf')

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=parameters,
        n_jobs = jobs,
        cv = 5,
        verbose = False)
  
    grid_search.fit(X_scaled_train, Y_scaled_train)

    svm_reg_rbf = svm.SVR(kernel='rbf', 
                C=grid_search.best_estimator_.C, # 2
                max_iter=grid_search.best_estimator_.max_iter, # 1000
                tol=grid_search.best_estimator_.tol, #0.0005
                gamma= grid_search.best_estimator_.gamma, # 1.6
                degree = grid_search.best_estimator_.degree) # 2

    svm_reg_rbf.fit(X_scaled_train, Y_scaled_train.ravel())
    preds_rbf = svm_reg_rbf.predict(X_scaled_test)

    preds_transformed2 = scaler_y.inverse_transform(preds_rbf.reshape(-1, 1))

    rmse2 = mean_squared_error(y_test, preds_transformed2, squared=False)
    print("Mean Squared Error: %f" % (rmse2))

    max2 = max_error(y_test, preds_transformed2)
    print("Max Error: %f" % (max2))

    MAE2 = median_absolute_error(y_test, preds_transformed2)
    print("Median Abs Error: %f" % (MAE2))

    x = datetime.datetime.now()

    d = {'target': ['gz', 'gz'],
        'Offset':[OS, OS],
        'RMSE': [rmse1, rmse2],
        'MAE': [MAE1, MAE2],
        'MaxError': [max1, max2], 
        'day': [x.day, x.day], 
        'month':[x.month, x.month], 
        'year':[x.year, x.year],
        'model':['SVM-RBF', 'SVM-RBF'],
        'version':[sklearn.__version__, sklearn.__version__]}

    results = pd.DataFrame(data=d)

    return results

def svm_perm(X_train, X_test, y_train, y_test, jobs, OS='os'):
    # This is a SVM, let's scale this data!
    n_feat = np.shape(X_train)[1]
  
    scaler_X = preprocessing.StandardScaler().fit(X_train, n_feat)
    scaler_y = preprocessing.StandardScaler().fit(y_train)

    X_scaled_train = scaler_X.transform(X_train)
    X_scaled_test  = scaler_X.transform(X_test) # use the same transformer
    Y_scaled_train = scaler_y.transform(y_train)

    svm_rbf = svm.SVR(kernel='rbf')
    svm_rbf.fit(X_scaled_train, Y_scaled_train)
    preds = svm_rbf.predict(X_scaled_test)

    preds_transformed = scaler_y.inverse_transform(preds)

    rmse1 = mean_squared_error(y_test, preds_transformed, squared=False)
    print("Mean Squared Error: %f" % (rmse1))
    max1 = max_error(y_test, preds_transformed)
    print("Max Error: %f" % (max1))
    MAE1 = median_absolute_error(y_test, preds_transformed)
    print("Median Abs Error: %f" % (MAE1))

    parameters = {
    'C': [ 300, 500, 750, 1000, 1250, 1500, 2000],
    'tol':[0.001, 0.01, 0.02, 0.4, 0.8],
    'gamma': [ 'auto', 'scale', 0.1, 1, 10],
    'max_iter': [25, 75, 150],
    'epsilon': [0.05, 0.1, 0.15, 0.2, 0.4],
    'degree': [1, 2,3,4,5]}

    estimator = svm.SVR(kernel='rbf')

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=parameters,
        n_jobs = jobs,
        cv = 5,
        verbose = False)
  
    grid_search.fit(X_train, y_train)

    svm_reg_rbf = svm.SVR(kernel='rbf', 
                      C=grid_search.best_estimator_.C, # 2
                      max_iter=grid_search.best_estimator_.max_iter, # 1000
                      tol=grid_search.best_estimator_.tol, #0.0005
                      gamma= grid_search.best_estimator_.gamma,
                      epsilon=grid_search.best_estimator_.epsilon, 
                      degree = grid_search.best_estimator_.degree) # 2

    svm_reg_rbf.fit(X_scaled_train, Y_scaled_train)
    preds_rbf = svm_reg_rbf.predict(X_scaled_test)

    preds_transformed2 = scaler_y.inverse_transform(preds_rbf)

    rmse2 = mean_squared_error(y_test, preds_transformed2, squared=False)
    print("Mean Squared Error: %f" % (rmse2))

    max2 = max_error(y_test, preds_transformed2)
    print("Max Error: %f" % (max2))

    MAE2 = median_absolute_error(y_test, preds_transformed2)
    print("Median Abs Error: %f" % (MAE2))

    x = datetime.datetime.now()

    d = {'target': ['perm', 'perm'],
    'Offset' :[OS, OS],
    'RMSE': [rmse1, rmse2],
    'MAE': [MAE1, MAE2],
    'MaxError': [max1, max2], 
    'day': [x.day, x.day], 
    'month':[x.month, x.month], 
    'year':[x.year, x.year],
    'model':['SVM-RBF', 'SVM-RBF'],
    'version':[sklearn.__version__, sklearn.__version__]}

    results = pd.DataFrame(data=d)

    return results

def multi_xrf_svm(X_train, X_test, y_train, y_test, OS='os'):
    # This is a SVM, let's scale this data!
    n_feat = np.shape(X_train)[1]
    n_feat_y = np.shape(y_train)[1]
  
    scaler_X = preprocessing.StandardScaler().fit(X_train, n_feat)
    scaler_y = preprocessing.StandardScaler().fit(y_train, n_feat_y)

    X_scaled_train = scaler_X.transform(X_train)
    X_scaled_test  = scaler_X.transform(X_test) # use the same transformer
    Y_scaled_train = scaler_y.transform(y_train)

    multioutputregressor = MultiOutputRegressor(svm.SVR(
                kernel='rbf', 
                C=5, 
                max_iter=800, 
                tol=0.01, 
                gamma= 0.1,
                degree = 2))

    multioutputregressor.fit(X_scaled_train, Y_scaled_train)

    preds = multioutputregressor.predict(X_scaled_test)
    preds_transformed = scaler_y.inverse_transform(preds)

    rmse2 = mean_squared_error(y_test,  preds_transformed, squared=False)
    print("Mean Squared Error: %f" % (rmse2))

    MAE2 = median_absolute_error(y_test,  preds_transformed)
    print("Median Abs Error: %f" % (MAE2))

    x = datetime.datetime.now()

    d = {'target': ['MultiXRF'],
        'Offset':[OS],
        'RMSE': [rmse2],
        'MAE': [MAE2],
        'day': [x.day], 
        'month':[x.month], 
        'year':[x.year],
        'model':['SVM-RBF'],
        'version':[sklearn.__version__]}

    results = pd.DataFrame(data=d)

    return results

def svm_por(X_train, X_test, y_train, y_test, jobs, OS='os'):
    # This is a SVM, let's scale this data!
    n_feat = np.shape(X_train)[1]
  
    scaler_X = preprocessing.StandardScaler().fit(X_train, n_feat)
    scaler_y = preprocessing.StandardScaler().fit(y_train)

    X_scaled_train = scaler_X.transform(X_train)
    X_scaled_test  = scaler_X.transform(X_test) # use the same transformer
    Y_scaled_train = scaler_y.transform(y_train)

    svm_rbf = svm.SVR(kernel='rbf')
    svm_rbf.fit(X_scaled_train, Y_scaled_train)
    preds = svm_rbf.predict(X_scaled_test)

    preds_transformed = scaler_y.inverse_transform(preds.reshape(-1, 1))

    rmse1 = mean_squared_error(y_test, preds_transformed, squared=False)
    print("Mean Squared Error: %f" % (rmse1))
    max1 = max_error(y_test, preds_transformed)
    print("Max Error: %f" % (max1))
    MAE1 = median_absolute_error(y_test, preds_transformed)
    print("Median Abs Error: %f" % (MAE1))

    parameters = {
        'C': [100, 200, 300, 400, 500, 750, 1000, 1250, 1500],
        'tol':[0.0005, 0.001, 0.003, 0.005, 0.01, 0.02, 0.4],
        'gamma': [ 'auto', 'scale', 0.1, 1, 3, 10],
        'max_iter': [25, 75, 150],
        'epsilon': [0.05, 0.1, 0.15, 0.2],
        'degree': [2,3,4,5]}

    estimator = svm.SVR(kernel='rbf')

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=parameters,
        n_jobs = jobs,
        cv = 5,
        verbose = False)
  
    grid_search.fit(X_scaled_train, Y_scaled_train)

    svm_reg_rbf = svm.SVR(kernel='rbf', 
                C=grid_search.best_estimator_.C, 
                max_iter=grid_search.best_estimator_.max_iter, 
                tol=grid_search.best_estimator_.tol, 
                gamma= grid_search.best_estimator_.gamma,
                epsilon= grid_search.best_estimator_.epsilon,
                degree = grid_search.best_estimator_.degree )

    svm_reg_rbf.fit(X_scaled_train, Y_scaled_train)
    preds_rbf = svm_reg_rbf.predict(X_scaled_test)

    preds_transformed2 = scaler_y.inverse_transform(preds_rbf.reshape(-1, 1))

    rmse2 = mean_squared_error(y_test, preds_transformed2, squared=False)
    print("Root Mean Squared Error: %f" % (rmse2))

    max2 = max_error(y_test, preds_transformed2)
    print("Max Error: %f" % (max2))

    MAE2 = median_absolute_error(y_test, preds_transformed2)
    print("Median Abs Error: %f" % (MAE2))

    x = datetime.datetime.now()

    d = {'target': ['por', 'por'],
        'Offset':[OS,OS],
        'MSE': [rmse1, rmse2],
        'MAE': [MAE1, MAE2],
        'MaxError': [max1, max2], 
        'day': [x.day, x.day], 
        'month':[x.month, x.month], 
        'year':[x.year, x.year],
        'model':['SVM-RBF', 'SVM-RBF'],
        'version':[sklearn.__version__, sklearn.__version__]}

    results = pd.DataFrame(data=d)

    return results