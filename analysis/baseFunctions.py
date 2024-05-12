import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from itertools import combinations
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA as ARIMA
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from scipy.signal import periodogram

#dec = sm.tsa.seasonal_decompose(dailyData,period = 12, model = 'additive').plot()

def addSeasonality(dt, dec, outputLength):
    first365Vals = dec.seasonal[0:dt] 
    first365Vals = first365Vals / max(first365Vals) 

    folds = int(outputLength/ dt)
    rest = (outputLength % dt) 

    seasonalVals = np.ones((outputLength)) * np.nan
    for i in range(folds):
        firstId = i*dt
        second = firstId + dt
        seasonalVals[firstId:second] = first365Vals
    seasonalVals[second:second+rest] = first365Vals[0:rest]

    return seasonalVals

def test_stationarity(timeseries, window = 12, printOutput=True):
    #Determing rolling statistics
    MA = timeseries.rolling(window = window).mean()
    MSTD = timeseries.rolling(window = window).std()

    #Plot rolling statistics:
    plt.figure(figsize=(15,5))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(MA, color='red', label='Rolling Mean')
    std = plt.plot(MSTD, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    if printOutput:
        print(dfoutput)
    return dftest[1] # return p value

def tsplot(y, lags=None, name = '', figsize=(12, 7), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        
        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
        plt.savefig(name + '_pacf.jpg')
        fig.show()

def seasonal_plot(X, y, period, freq, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    palette = sns.color_palette("husl", n_colors=X[period].nunique(),)
    ax = sns.lineplot(
        x=freq,
        y=y,
        hue=period,
        data=X,
        ci=False,
        ax=ax,
        palette=palette,
        legend=False,
    )
    ax.set_title(f"Seasonal Plot ({period}/{freq})")
    for line, name in zip(ax.lines, X[period].unique()):
        y_ = line.get_ydata()[-1]
        ax.annotate(
            name,
            xy=(1, y_),
            xytext=(6, 0),
            color=line.get_color(),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            size=14,
            va="center",
        )
    return ax


def plot_periodogram(ts, cyclesPerUnit, n_domFreq = 15, detrend='linear', ax=None):
    freqencies, power = periodogram(
        ts,
        fs=cyclesPerUnit,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, power, color="purple")
    ax.set_xscale("log")
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")

    sorted_indices = np.argsort(power)[::-1]  # Sort indices in descending order of power
    dominant_freqs = freqencies[sorted_indices[:n_domFreq]]
    print(dominant_freqs)
    #return freqencies, power 


def addFourierFeature(data1, n_splits, frequency, feature, referenceTimespan):
    # data1                 dataframe that will be returned
    # n_splits              how many fourier features per frequency do we want
    # frequency             frequency of feature based on reference timespan, e.g. 365 days, frequency of 12 = monthly = 12 sinus periods per year, frequency of 1 = 1 sinus curve per year
    # feature               feature we want to base the feature on
    # referenceTimespan     timespan we base our frequency on, e.g. 365 days for a year (if I have daily data)
    periodicfeat = []
    for i in range(n_splits):
        deg = int(360/n_splits *i)
        f = feature +'_f'+str(frequency)+'_'+str(deg)
        periodicfeat.append(f)
        data1[f] = np.sin(2*np.pi*(frequency*data1[feature]/referenceTimespan) + 2*np.pi* i/n_splits)
    return data1, periodicfeat

# resample stuff
# df_resampled = ePrices.resample('1H').asfreq()