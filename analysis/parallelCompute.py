import tensorflow as tf
tf.random.set_seed(42)
import numpy as np
np.random.seed(42)
import time
import pickle

import pandas as pd

from datetime import datetime

import statsmodels.api as sm
import statsmodels.tsa.api as smt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import lightgbm as lgb

from datetime import datetime, timedelta
import multiprocessing


from baseFunctions import *
from data_helpers import processData6, featureEngineering, getSequencesFast, removeOutliers, create_sequences,dataProcessing


def calcLossLGBM(pred, y, logTransform, predictDiff, base):
    if predictDiff:
        pred = np.reshape(pred, (y.shape[0])) +  np.reshape(base, (y.shape[0]))
        y = np.reshape(y, (y.shape[0])) + np.reshape(base, (y.shape[0]))

    if logTransform:
        a = np.exp(pred) -1
        y = np.exp(y) -1 
    else:
        a = (pred)
        y = (y)

    if (a < 0).any():
        a = np.clip(a, 0, 1e20)

    rmsleTrain = np.sqrt(mean_squared_log_error(a,y))
    return rmsleTrain


def fitLGBM(args):
       storeDf, date_string_val, date_string_test, predictDiff, seasonalFDiff, seasonalLags, targetLags, rolling, initial_lag = args
       train_subDf, test_subDf, pred_subDf, allF = dataProcessing(storeDf, date_string_val, date_string_test, predictDiff, seasonalFDiff, seasonalLags, targetLags, rolling, initial_lag, prepPred = True)
       
       targetF = 'target'
       if predictDiff:
              baseTrain = train_subDf['ref'].to_numpy()
              baseTest  = test_subDf['ref'].to_numpy()
              baseVal   = pred_subDf['ref'].to_numpy()
       else:
              baseTrain, baseTest, baseVal = [],[],[]

       X_train = train_subDf[allF].to_numpy()
       y_train = train_subDf[targetF].to_numpy()
       X_test  =  test_subDf[allF].to_numpy()
       y_test  =  test_subDf[targetF].to_numpy()  
       X_pred   =  pred_subDf[allF].to_numpy()

       params = {
           'boosting':'gbdt',#'gbdt', #'rf' #'dart'
           'objective': 'regression',  # Assuming you're doing regression
           'metric': 'mse',  # Mean squared error
           'num_leaves': 5,
           'feature_fraction': 0.9,
           'bagging_fraction': 0.8,
           'bagging_freq': 5,
           'verbose': -1,
           'force_col_wise':True,
           #'num_iterations':500
       }   

       # Train the model
       gbm = lgb.train(params, lgb.Dataset(X_train, label=y_train),500, valid_sets=[lgb.Dataset(X_test, label=y_test)],callbacks=[lgb.early_stopping(stopping_rounds=100)])
       predtrainLGBM = gbm.predict(X_train)
       predtestLGBM = gbm.predict(X_test)
       predLGBM = gbm.predict(X_pred)

       logTransform = True
       trainLoss = calcLossLGBM(predtrainLGBM, y_train, logTransform, predictDiff, base=baseTrain)
       testLoss  = calcLossLGBM(predtestLGBM, y_test, logTransform, predictDiff, baseTest)
       storeId = storeDf.store_nbr.unique()[0]
       family = storeDf.family.unique()[0]
       print('store',storeId,'family',family,'errors: ', trainLoss, testLoss,gbm.best_iteration)

       subDf = pred_subDf[['id','sales']]
       subDf.loc[:,['sales']] = np.exp(predLGBM)-1
       subDf.loc[:,['trainE']] = trainLoss
       subDf.loc[:,['testE']] = testLoss
       subDf.loc[:,['bestIt']] = gbm.best_iteration

       return subDf




if __name__ == '__main__':

    data, propDicts, flippedPropDicts = processData6(oneHotEnodeCatFeat=True)
    data, timeFeatures = featureEngineering(data,splits=[2,2,2,2],oneHotWeekday=True)

      # helps to learn function, but doesn't help with overfitting
    data['linear_time'] = (data['linear_time'] - data.linear_time.mean()) /data.linear_time.std()
    data['day_of_year'] = (data['day_of_year'] - data.day_of_year.mean()) /data.day_of_year.std()
    data['dcoilwtico'] = (data['dcoilwtico'] - data.dcoilwtico.mean()) /data.dcoilwtico.std()
    data['transactions'] = (data['transactions'] - data.transactions.mean()) /data.transactions.std()

    

    initial_lag = 16 # 16 = independent from previous predictions
    seasonalFDiff = 1
    seasonalLags=[1,2,3,4,5,6,7,14,21,52,104]#range(21)
    targetLags = [0,1,2,3,4,5,6,7,14,21,52,10]#range(21)# [7,14,21,28,35,42,52] #range(52)#[7,14,21]
    rolling = [7,14,21,28,35,42,52,104]
    predictDiff =False
    logTransform=True

    # Date string
    date_string_val = "2017-07-15"#"2017-05-01"
    date_string_test = "2017-07-31"

    pool = multiprocessing.Pool(6)

    input = []
    print('done preprocesing')
    for familyId in data.family.unique():
           # start with only some families
           print('processing',familyId)
           familyDf = data.loc[data.family==familyId]  

           for storeId in data.store_nbr.unique():
                  storeDf = familyDf.loc[(familyDf.store_nbr == storeId)] 

                  input.append([storeDf, date_string_val, date_string_test, predictDiff, seasonalFDiff, seasonalLags, targetLags, rolling, initial_lag])
    print('done appending')

    results = list(pool.map(fitLGBM, input)) 
    print(results)

    df = pd.concat(results, axis = 0)
    df.to_csv('multiprocess_lgbm.csv')