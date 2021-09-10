"""
Catboost functions for regression


"""

import pandas as pd # data storage
import catboost as cats # graident boosting 

from catboost import CatBoostRegressor, Pool

import datetime
import numpy as np  # math and stuff
import matplotlib.pyplot as plt # plotting utility
import sklearn # ML and stats

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import max_error, mean_squared_error, median_absolute_error

#### Look below here

def catboost_gz(X_train, X_test, y_train, y_test, max_iter=500, CPUorGPU='CPU', OS='os', target='gz'):

  model = CatBoostRegressor(objective='RMSE',
                            task_type=CPUorGPU,
                            iterations=max_iter)
  
  model.fit(X_train, y_train, verbose=1000 )
  preds = model.predict(X_test)

  # Stats
  rmse = mean_squared_error(y_test, preds, squared=False)
  print("Root Mean Squared Error: %f" % (rmse))
  max = max_error(y_test, preds)
  print("Max Error: %f" % (max))
  MAE= median_absolute_error(y_test, preds)
  print("Median Abs Error: %f" % (max))

  # Let's Tune
  grid = {'learning_rate': [0.1, 0.2, 0.4],
        'depth': [6, 8, 10,  12, 14],
        'l2_leaf_reg': [2, 3, 4 , 5, 6 ]}

  model_grid = CatBoostRegressor(objective='RMSE',
                                 task_type=CPUorGPU,
                                 iterations=max_iter)

  # Grid Search
  grid_search_result = model_grid.grid_search(grid, 
                                            X=X_train, 
                                            y=y_train, 
                                            cv=5,
                                            verbose=False)
  model2 = CatBoostRegressor(objective='RMSE',
                           task_type=CPUorGPU,
                           depth=grid_search_result['params']['depth'],
                           l2_leaf_reg=grid_search_result['params']['l2_leaf_reg'],
                           learning_rate=grid_search_result['params']['learning_rate'],
                           iterations=max_iter)

  model2.fit(X_train, y_train, verbose=500)
  preds2 = model2.predict(X_test)

  # Stats for the second round
  rmse2 = mean_squared_error(y_test, preds2, squared=False)
  print("Root Mean Squared Error: %f" % (rmse2))
  max2 = max_error(y_test, preds2)
  print("Max Error: %f" % (max2))
  MAE2= median_absolute_error(y_test, preds2)
  print("Median Abs Error: %f" % (MAE2))

  x = datetime.datetime.now()

  d = {'target': [target, target],
      'offset': [OS, OS],
     'RMSE': [rmse, rmse2],
     'MAE': [MAE, MAE2],
     'MaxError': [max, max2], 
     'iter': [max_iter, max_iter],
     'day': [x.day, x.day], 
     'month':[x.month, x.month], 
     'year':[x.year, x.year],
     'model':['catboost', 'catboost'],
     'version':[cats.__version__, cats.__version__ ]}

  results = pd.DataFrame(data=d)

  return results


def catboost_multi_xrf(X_train, X_test, y_train, y_test, max_iter=1500, OS='os'):

  model = CatBoostRegressor(objective='MultiRMSE', iterations=max_iter)
  model.fit(X_train, y_train, verbose=500 )

  preds = model.predict(X_test)

  rmse = mean_squared_error(y_test, preds, squared=False)
  print("Root Mean Squared Error: %f" % (rmse))
  MAE = median_absolute_error(y_test, preds)
  print("Median Abs Error: %f" % (MAE))

  grid = {'learning_rate': [0.05, 0.1, 0.2, 0.3], # 0.1
        'depth': [ 4, 6, 8], #6
        'l2_leaf_reg': [3, 4, 5, 6]}

  model_grid = CatBoostRegressor(objective='MultiRMSE',
                                 verbose=False,
                                 iterations=max_iter)

  # Grid Search
  grid_search_result = model_grid.grid_search(grid, 
                                            X=X_train, 
                                            y=y_train, 
                                            cv=5,
                                            verbose=False)
  
  model2 = CatBoostRegressor(objective='MultiRMSE',
                           depth=grid_search_result['params']['depth'],
                           l2_leaf_reg=grid_search_result['params']['l2_leaf_reg'],
                           learning_rate=grid_search_result['params']['learning_rate'],
                           iterations=max_iter)

  model2.fit(X_train, y_train, verbose=500 )

  preds2 = model2.predict(X_test)

  rmse2 = mean_squared_error(y_test, preds2, squared=False)
  print("Root Mean Squared Error: %f" % (rmse2))

  MAE2 = median_absolute_error(y_test, preds2)
  print("Median Abs Error: %f" % (MAE2))

  x = datetime.datetime.now()

  d = {'target': ['MultiXRF', 'MultiXRF'],
     'offset': [OS, OS],
     'RMSE': [rmse, rmse2],
     'MAE': [MAE, MAE2],
     'iter': [max_iter, max_iter],
     'day': [x.day, x.day], 
     'month':[x.month, x.month], 
     'year':[x.year, x.year],
     'model':['catboost', 'catboost'],
     'version':[cats.__version__, cats.__version__ ]}

  results = pd.DataFrame(data=d)

  return results


def catboost_perm(X_train, X_test, y_train, y_test, max_iter = 200, CPUorGPU='CPU', OS='os'):
  model = CatBoostRegressor(objective='RMSE',
                            task_type=CPUorGPU,
                            iterations=max_iter)

  model.fit(X_train, y_train, verbose=max_iter )

  preds = model.predict(X_test)

  rmse = mean_squared_error(y_test, preds, squared=False)
  print("Root Mean Squared Error: %f" % (rmse))
  max = max_error(y_test, preds)
  print("Max Error: %f" % (max))
  MAE = median_absolute_error(y_test, preds)
  print("Median Abs Error: %f" % (MAE))

  grid = {'learning_rate': [ 0.05, 0.1, 0.2, 0.3],
        'depth': [ 4, 6, 8, 10],
        'l2_leaf_reg': [ 3, 4, 5, 6, 7, 8]}

  model_grid = CatBoostRegressor(objective='RMSE', 
                                 iterations=max_iter, 
                                 verbose=False)

  # Grid Search
  grid_search_result = model_grid.grid_search(grid, 
                                            X=X_train, 
                                            y=y_train, 
                                            cv=5,
                                            verbose=False)
  
  model2 = CatBoostRegressor(objective='RMSE',
                           task_type=CPUorGPU,
                           depth=grid_search_result['params']['depth'],
                           l2_leaf_reg=grid_search_result['params']['l2_leaf_reg'],
                           learning_rate=grid_search_result['params']['learning_rate'],
                           iterations=max_iter)

  model2.fit(X_train, y_train, verbose=500 )

  preds2 = model2.predict(X_test)

  rmse2 = mean_squared_error(y_test, preds2, squared=False)
  print("Root Mean Squared Error: %f" % (rmse2))
  max2 = max_error(y_test, preds2)
  print("Max Error: %f" % (max2))
  MAE2= median_absolute_error(y_test, preds2)
  print("Median Abs Error: %f" % (MAE2))

  x = datetime.datetime.now()

  d = {'target': ['perm', 'perm'],
     'offset': [OS, OS],
     'RMSE': [rmse, rmse2],
     'MAE': [MAE, MAE2],
     'MaxError': [max, max2], 
     'iter':[max_iter, max_iter],
     'day': [x.day, x.day], 
     'month':[x.month, x.month], 
     'year':[x.year, x.year],
     'model':['catboost', 'catboost'],
     'version':[cats.__version__, cats.__version__ ]}

  results = pd.DataFrame(data=d)

  return results


def catboost_he_por(X_train, X_test, y_train, y_test, OS='os', max_iter = 200, CPUorGPU='CPU'):
  model = CatBoostRegressor(objective='RMSE', 
                            task_type=CPUorGPU,
                            iterations=max_iter)

  model.fit(X_train, y_train, verbose=max_iter )

  preds = model.predict(X_test)

  rmse = mean_squared_error(y_test, preds, squared=False)
  print("Root Mean Squared Error: %f" % (rmse))
  max = max_error(y_test, preds)
  print("Max Error: %f" % (max))
  MAE = median_absolute_error(y_test, preds)
  print("Median Abs Error: %f" % (MAE))

  grid = {'learning_rate': [ 0.05, 0.1, 0.2, 0.3],
        'depth': [ 2, 4, 6, 8, 10],
        'l2_leaf_reg': [1, 2, 3, 4, 5, 7]}

  model_grid = CatBoostRegressor(objective='RMSE', 
                                 task_type=CPUorGPU,
                                 iterations=max_iter, 
                                 verbose=False)

  # Grid Search
  grid_search_result = model_grid.grid_search(grid, 
                                            X=X_train, 
                                            y=y_train, 
                                            cv=5,
                                            verbose=False)
  
  model2 = CatBoostRegressor(objective='RMSE',
                        task_type=CPUorGPU,
                        depth=grid_search_result['params']['depth'],
                        l2_leaf_reg=grid_search_result['params']['l2_leaf_reg'],
                        learning_rate=grid_search_result['params']['learning_rate'],
                        iterations=max_iter)

  model2.fit(X_train, y_train, verbose=500 )

  preds2 = model2.predict(X_test)

  rmse2 = mean_squared_error(y_test, preds2, squared=False)
  print("Root Mean Squared Error: %f" % (rmse2))
  max2 = max_error(y_test, preds2)
  print("Max Error: %f" % (max2))
  MAE2= median_absolute_error(y_test, preds2)
  print("Median Abs Error: %f" % (MAE2))

  x = datetime.datetime.now()

  d = {'target': ['poro', 'poro'],
     'offset': [OS, OS],
     'RMSE': [rmse, rmse2],
     'MAE': [MAE, MAE2],
     'MaxError': [max, max2], 
     'iter':[max_iter, max_iter],
     'day': [x.day, x.day], 
     'month':[x.month, x.month], 
     'year':[x.year, x.year],
     'model':['catboost', 'catboost'],
     'version':[cats.__version__, cats.__version__ ]}

  results = pd.DataFrame(data=d)

  return results