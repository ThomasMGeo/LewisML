# File for XGBoost ML functions

import pandas as pd # data storage
import xgboost as xgb # graident boosting 
import numpy as np  # math and stuff
import matplotlib.pyplot as plt # plotting utility
import sklearn # ML and stats

import datetime

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import  max_error, mean_squared_error, median_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV

def xgb_gz(X_train, X_test, y_train, y_test, OS='os', tree_type = 'auto'):
  xg_reg = xgb.XGBRegressor()

  xg_reg.fit(X_train,y_train)

  preds = xg_reg.predict(X_test)

  rmse = mean_squared_error(y_test, preds, squared=False)
  print("Root Mean Squared Error: %f" % (rmse))
  max = max_error(y_test, preds)
  print("Max Error: %f" % (max))
  MAE = median_absolute_error(y_test, preds)
  print("Median Abs Error: %f" % (MAE))

  parameters = {
    'max_depth': range (4, 8, 1),
    'n_estimators': range(50, 300, 25),
    'colsample_bytree':[ 0.6, 0.8, 0.9, 1],
    'learning_rate': [ 0.5, 0.3, 0.2, 0.1]}

  estimator = xgb.XGBRegressor(tree_method=tree_type,
                              gpu_id=-1) # uses the GPU

  # Grid Search
  grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    n_jobs = 8,
    cv = 5,
    verbose = 1)

  grid_search.fit(X_train, y_train)

  model1 = xgb.XGBRegressor(n_estimators=grid_search.best_estimator_.n_estimators, 
                          max_depth = grid_search.best_estimator_.max_depth,
                          learning_rate=grid_search.best_estimator_.learning_rate, 
                          colsample_bytree=grid_search.best_estimator_.colsample_bytree)

  model1.fit(X_train, y_train)

  preds2 = model1.predict(X_test)

  rmse2 = mean_squared_error(y_test, preds2, squared=False)
  print("Mean Squared Error: %f" % (rmse2))
  max1 = max_error(y_test, preds2)
  print("Max Error: %f" % (max1))
  MAE2 = median_absolute_error(y_test, preds2)
  print("Median Abs Error: %f" % (MAE2))

  x = datetime.datetime.now()

  d = {'target': ['gz', 'gz'],
     'offset': [OS, OS],
     'RMSE': [rmse, rmse2],
     'MAE': [MAE, MAE2],
     'MaxError': [max, max1], 
     'day': [x.day, x.day], 
     'month':[x.month, x.month], 
     'year':[x.year, x.year],
     'model':['XGB', 'XGB'],
     'version':[xgb.__version__, xgb.__version__ ]}

  results = pd.DataFrame(data=d)

  return results

def xgb_perm(X_train, X_test, y_train, y_test, OS='os', tree_type='hist' ):
  xg_reg = xgb.XGBRegressor(tree_method=tree_type)

  xg_reg.fit(X_train,y_train)

  preds = xg_reg.predict(X_test)

  rmse = mean_squared_error(y_test, preds, squared=False)
  print("Root Mean Squared Error: %f" % (rmse))
  max = max_error(y_test, preds)
  print("Max Error: %f" % (max))
  MAE = median_absolute_error(y_test, preds)
  print("Median Abs Error: %f" % (MAE))

  parameters = {
    'max_depth': range (2, 6, 1),
    'n_estimators': range(20, 140, 20),
    'colsample_bytree': [0.7, 0.8, 0.9, 1],
    'learning_rate': [ 0.3, 0.2, 0.1, 0.05],
    'max_delta_step':  [ 0, 1],
    'reg_alpha' : [0, 1]}

  estimator = xgb.XGBRegressor(tree_method=tree_type, gpu_id=-1) # uses the GPU

  # Grid Search
  grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    n_jobs = 8,
    cv = 5,
    verbose = 1)

  grid_search.fit(X_train, y_train)

  model1 = xgb.XGBRegressor(tree_method=tree_type,
                          n_estimators=grid_search.best_estimator_.n_estimators, 
                          max_depth = grid_search.best_estimator_.max_depth,
                          learning_rate=grid_search.best_estimator_.learning_rate, 
                          colsample_bytree=grid_search.best_estimator_.colsample_bytree)

  model1.fit(X_train, y_train)

  preds2 = model1.predict(X_test)

  rmse2 = mean_squared_error(y_test, preds2, squared=False)
  print("Mean Squared Error: %f" % (rmse2))
  max1 = max_error(y_test, preds2)
  print("Max Error: %f" % (max1))
  MAE2 = median_absolute_error(y_test, preds2)
  print("Median Abs Error: %f" % (MAE2))

  x = datetime.datetime.now()

  d = {'target': ['perm', 'perm'],
     'Offset': [OS, OS],
     'RMSE': [rmse, rmse2],
     'MAE': [MAE, MAE2],
     'MaxError': [max, max1], 
     'day': [x.day, x.day], 
     'month':[x.month, x.month], 
     'year':[x.year, x.year],
     'model':['XGB', 'XGB'],
     'version':[xgb.__version__, xgb.__version__ ]}

  results = pd.DataFrame(data=d)

  return results

def xgb_multi_xrf(X_train, X_test, y_train, y_test, OS='os'):
  multioutputregressor = MultiOutputRegressor(xgb.XGBRegressor(
            tree_method = 'hist',
            objective ='reg:squarederror', 
            colsample_bytree = 0.9, 
            learning_rate = 0.1, 
            max_depth = 5, 
            n_estimators = 60))

  multioutputregressor.fit(X_train,y_train)

  preds = multioutputregressor.predict(X_test)

  rmse = mean_squared_error(y_test, preds, squared=False)
  print("Root Mean Squared Error: %f" % (rmse))

  MAE = median_absolute_error(y_test, preds)
  print("Median Abs Error: %f" % (MAE))

  x = datetime.datetime.now()

  d = {'target': ['MultiXRF'],
  'Offset':[OS],
     'MSE': [rmse],
     'MAE': [MAE ], 
     'day': [x.day], 
     'month':[x.month], 
     'year':[x.year],
     'model':['XGB'],
     'version':[xgb.__version__]}

  results = pd.DataFrame(data=d)

  return results

def xgb_por(X_train, X_test, y_train, y_test, OS='os', tree_type='auto'):
    xg_reg = xgb.XGBRegressor()

    xg_reg.fit(X_train,y_train)

    preds = xg_reg.predict(X_test)

    rmse = mean_squared_error(y_test, preds, squared=False)
    print("Root Mean Squared Error: %f" % (rmse))
    max = max_error(y_test, preds)
    print("Max Error: %f" % (max))
    MAE = median_absolute_error(y_test, preds)
    print("Median Abs Error: %f" % (MAE))

    parameters = {
    'max_depth': range (4, 6, 1),
    'n_estimators': range(20, 55, 5),
    'colsample_bytree': [ 0.9, 1],
    'learning_rate': [ 0.2, 0.1, 0.05],
    'max_delta_step':  [ 0, 1, 2, 4, 8, 16],
    'reg_alpha' : [0, 1, 2, 4]
    }

    estimator = xgb.XGBRegressor(tree_method=tree_type)
                             
    # Grid Search
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=parameters,
        n_jobs = 8,
        cv = 5,
        verbose = 1)

    grid_search.fit(X_train, y_train)

    model1 = xgb.XGBRegressor(
                        tree_method=tree_type,
                        n_estimators=grid_search.best_estimator_.n_estimators, 
                        max_depth = grid_search.best_estimator_.max_depth,
                        learning_rate=grid_search.best_estimator_.learning_rate, 
                        colsample_bytree=grid_search.best_estimator_.colsample_bytree
                        )

    model1.fit(X_train, y_train)

    preds2 = model1.predict(X_test)

    rmse2 = mean_squared_error(y_test, preds2, squared=False)
    print("Mean Squared Error: %f" % (rmse2))
    max1 = max_error(y_test, preds2)
    print("Max Error: %f" % (max1))
    MAE2 = median_absolute_error(y_test, preds2)
    print("Median Abs Error: %f" % (MAE2))

    x = datetime.datetime.now()

    d = {'target': ['por', 'por'],
     'offset': [OS, OS],
     'RMSE': [rmse, rmse2],
     'MAE': [MAE, MAE2],
     'MaxError': [max, max1], 
     'day': [x.day, x.day], 
     'month':[x.month, x.month], 
     'year':[x.year, x.year],
     'model':['XGB', 'XGB'],
     'version':[xgb.__version__, xgb.__version__ ]}

    results = pd.DataFrame(data=d)

    return results