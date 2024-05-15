import tensorflow as tf
tf.random.set_seed(42)
import numpy as np
np.random.seed(42)
import time
import pickle

import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from statsmodels.tsa.statespace.sarimax import SARIMAX

from datetime import datetime, timedelta
import multiprocessing


from baseFunctions import *
from data_helpers import processData6, featureEngineering, getSequencesFast, removeOutliers, create_sequences


def calcLossArima(fitted, y_train, logTransform):
    if not logTransform:
        rmsleTrain = np.sqrt(mean_squared_log_error(fitted,y_train))
    else:
        y_train = np.reshape(y_train, fitted.shape)
        rmsleTrain =  np.sqrt(np.mean((fitted-y_train)**2))
    return rmsleTrain

def predictSarima(params):
    storeDf = params

    #train = storeDf.loc[storeDf.dataT =='train']
    pred = storeDf.loc[storeDf.dataT =='test']
    #date_string_test = "2017-07-31"
    #train = storeDf.loc[storeDf.date < date_string_test].set_index('date')
    #test = storeDf.loc[storeDf.date >= date_string_test].set_index('date')
    train = storeDf.loc[storeDf.dataT =='train'].set_index('date')
    train = train.asfreq('D')
    #test = test.asfreq('D')

    y_trainSarima = np.log(train.sales+1)

    model = SARIMAX(y_trainSarima, order=(5, 1, 5), seasonal_order=(3,1,3,7))
    model_fit = model.fit(disp=0)
    fitted_values = model_fit.fittedvalues
    lossTrain = calcLossArima(fitted_values, y_trainSarima, True)

    predicted_values = model_fit.forecast(steps=16)
    #predicted_values = model_fit.forecast(steps=test.shape[0] + 16)
    
    # test_pred = predicted_values[0:test.shape[0]]
    # lossTest  = calcLossArima(test_pred, y_testSarima, True)

    predicted_values = np.reshape(np.exp(predicted_values)-1, (-1,1))
    pred.loc[:,['sales']] = predicted_values#[test.shape[0]:test.shape[0]+16]
    

    storeId = storeDf.store_nbr.unique()[0]
    family = storeDf.family.unique()[0]
    print('store',storeId,'family',family,'errors: ', lossTrain)#, lossTest)

    subDf = pred[['id','sales']]
    #subDf.loc[:,['testE']] = lossTest
    subDf.loc[:,['trainE']] = lossTrain
    return subDf





if __name__ == '__main__':
    data, propDicts, flippedPropDicts = processData6()
    data, timeFeatures = featureEngineering(data,splits=[2,2,2,2])

    pool = multiprocessing.Pool(6)

    input = []
    for familyId in data.family.unique():
        print('----family ----',familyId)
        familyDf = data.loc[data.family==familyId]

        for storeId in data.store_nbr.unique():
            #print('----store ----',storeId)
            storeDf = familyDf.loc[(familyDf.store_nbr == storeId)]# & (storeDf.date > "2015-07-01")]
            input.append(storeDf)

    results = list(pool.map(predictSarima, input)) 

    df = pd.concat(results, axis = 0)
    df.to_csv('multiprocess_sarima515_3137.csv')
