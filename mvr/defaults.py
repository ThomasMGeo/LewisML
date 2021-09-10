import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

def framecleaner(df, data, inputz, tar):
    df = df.drop(['Unnamed: 0', 'Unnamed: 0.1', 'LiveTime2','ScanTime2', 'LiveTime1','ScanTime1',
              'ref_num', 'API', 'well_name', 'sample_num' ], axis=1)

    #df = df.replace('NaN',np.nan, regex=True, inplace=True) # Should be good already

    dataset = df[data]
    
    dataset.dropna(axis=0, inplace=True)
    # Features we will use for prediction
    X = dataset[inputz]

    # What we are trying to predict
    Y = dataset[tar]

    Y_array = np.array(Y.values)

    return X, Y_array

def splitterz(x, y, seedz=42, withhold=0.25):
    seed = seedz # random seed is only used if you want to compare exact answers with friends 

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=withhold)
    return X_train, X_test, y_train, y_test