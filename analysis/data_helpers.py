import pandas as pd
from baseFunctions import *

def processData6(oneHotEnodeCatFeat=False):
    # prepare data
    train = pd.read_csv('../train.csv')
    oil = pd.read_csv('../oil.csv')
    stores = pd.read_csv('../stores.csv')
    transactions = pd.read_csv('../transactions.csv')
    test = pd.read_csv('../test.csv')
    holidays = pd.read_csv('../holidays_events.csv')
    sampleSub = pd.read_csv('../sample_submission.csv')

    train['dataT'] = 'train'
    test['dataT'] = 'test'
    train['date'] = pd.to_datetime(train['date'])
    test['date'] = pd.to_datetime(test['date'])
    data = pd.concat([train, test])
    data['date'] = pd.to_datetime(data['date']) 

    data0 = pd.merge(data, stores, on=['store_nbr'], how='outer')
    print(data.shape, data0.shape)  

    oil['date'] = pd.to_datetime(oil['date'])
    oil.set_index('date',inplace=True)
    oil_resampled = oil.resample('1D').asfreq()
    print(oil_resampled.isna().sum())
    oil_resampled.interpolate(inplace=True,limit_direction='both')
    oil_resampled.reset_index(inplace=True)
    print(oil_resampled.isna().sum())   

    data0['date'] = pd.to_datetime(data0['date'])
    data1 = pd.merge(data0, oil_resampled, on=['date'], how='left')
    print(data1.shape, data0.shape) 

    holidays['date'] = pd.to_datetime(holidays['date'])
    cityHolidays = holidays.loc[holidays.locale =='Local']#.locale_name.value_counts()
    cityHolidays.drop('locale', axis = 1, inplace=True)
    cityHolidays['description'] = 'Fundacion'
    cityHolidays.rename(columns={'locale_name':'city','type':'holidayType'}, inplace=True)
    cityHolidays.drop(264, axis = 0, inplace=True) # we have some duplicates
    data2 = pd.merge(data1, cityHolidays, on=['date','city'], how='left')
    #train2.dropna(inplace=True)
    print(data1.shape, data2.shape,'data2') 

    regionalHolidays = holidays.loc[holidays.locale =='Regional']#.locale_name.value_counts()
    regionalHolidays.drop('locale', axis = 1, inplace=True)
    regionalHolidays['description'] = 'Provincializacion'
    regionalHolidays.rename(columns={'locale_name':'state'}, inplace=True)  

    data3 = pd.merge(data2, regionalHolidays, on=['date','state'], how='left', suffixes=('','_reg'))
    print(data3.shape, data2.shape,'data3') 

    nationalHolidays = holidays.loc[holidays.locale =='National']#.locale_name.value_counts()
    nationalHolidays.drop(['locale','locale_name'], axis = 1, inplace=True)
    nationalHolidays.description.unique()
    groups = ['Navidad', 'Mundial de futbol Brasil','Terremoto Manabi','dia del ano','Puente Dia de Difuntos','Grito de Independencia','Independencia de Guayaquil','Dia de la Madre','Batalla de Pichincha']
    for group in groups:
        mask = nationalHolidays['description'].str.contains(group)
        nationalHolidays.loc[mask, 'description'] = group
    nationalHolidays = nationalHolidays.drop_duplicates(subset=['date'], keep='first')  

    data4 = pd.merge(data3, nationalHolidays, on=['date'], how='left', suffixes=('','_nat'))
    print(data3.shape, data4.shape) 
    print('finished data4')

    data5 = data4.copy()
    data5['holidayType'] = data5['holidayType'].combine_first(data5['type_reg'])
    data5['holidayType'] = data5['holidayType'].combine_first(data5['type_nat'])
    data5['holidayType'] = data5['holidayType'].fillna(0)

    data5['description'] = data5['description'].combine_first(data5['description_reg'])
    data5['description'] = data5['description'].combine_first(data5['description_nat']) 
    data5['description'] = data5['description'].fillna(0)

    data5['transferred'] = data5['transferred'].combine_first(data5['transferred_reg'])
    data5['transferred'] = data5['transferred'].combine_first(data5['transferred_nat']) 
    data5['transferred'] = data5['transferred'].fillna(0)

    data5 = data5.drop(columns=['type_reg','type_nat','description_reg','description_nat','transferred_reg','transferred_nat']) 

    print(data4.shape, data5.shape) 
    print('finished data5')
    data6 = data5.copy()
    propDicts = {}
    for f in ['family','city','state','type','holidayType','description','transferred']:
        a = data6[f].value_counts().to_dict()
        unique = []
        # sort based on occurance
        for key, val in enumerate(a):
            unique.append(val)
        category_dict = {category: index for index, category in enumerate(unique)}
        data6[f] = data6[f].map(category_dict)
        propDicts[f] = category_dict    

    flippedPropDicts = {}
    for key,value in propDicts.items():
        flippedPropDicts[key] = {value: key for key, value in propDicts[key].items()}

    if oneHotEnodeCatFeat:
        for f in ['family','city','state','type','holidayType','description']:
            one_hot_encoded = pd.get_dummies(data6[f], prefix=f).astype(int)
            data6 = pd.concat([data6, one_hot_encoded], axis=1)
            #data6 = data6.drop('weekday', axis = 1)


    transactions['date'] = pd.to_datetime(transactions['date'])
    data7 =pd.merge(data6,transactions, on=['store_nbr','date'], how='left')

    data7['store_closed'] = data7.transactions.isna().astype(int)
    data7.loc[data7.dataT == 'test', 'store_closed'] = 0
    #data7 = data7.drop('transactions', axis = 1)

    return data7, propDicts, flippedPropDicts

def featureEngineering(data1, frequencies = [12,104,24,52], splits=[1,1,1,1], refTimeSpan=365, feature='day_of_year', oneHotWeekday = False):
    # add linear time
    data1['linear_time'] = (data1['date'] - data1['date'].iloc[0]).dt.days +1
    data1['day_of_year'] = data1['date'].dt.day_of_year

    featureNames = []

    for i,f in enumerate(frequencies):
        data1, periodicfeat = addFourierFeature(data1, n_splits = splits[i], frequency=f, feature=feature, referenceTimespan = refTimeSpan)
        featureNames = featureNames + periodicfeat

    data1['weekday'] = data1['date'].dt.weekday
    data1['month'] = data1['date'].dt.month

    if oneHotWeekday:
        one_hot_encoded = pd.get_dummies(data1.weekday, prefix='weekday').astype(int)
        data1 = pd.concat([data1, one_hot_encoded], axis=1)
        data1 = data1.drop('weekday', axis = 1)

    return data1, featureNames

def plotSales(df, storeId : int, family :str, familyId, p_value, save=False):
    plt.figure(figsize=(15,20))

    family=family.replace('/','-')
    
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 6))
    axs[0].plot(df.date, df.sales, color='blue',label='Original')
    axs[0].plot(df.date, df.sales_outRem, color='red',label='out_rem')
    axs[0].set_title(str(storeId)+'  '+str(familyId) + family + ' p_value:' + str(p_value))
    

    axs[1].plot(df.date, np.log(df.sales+1), color='blue')
    axs[1].set_title('log sales')
    axs[2].plot(df.date ,np.log(df.sales+1).diff(21), color='blue')
    axs[2].set_title('log sales, diff 3 weeks')

    fig.subplots_adjust(hspace=0.5)
    #orig = plt.plot(df.sales, color='blue',label='Original')
    #orig = plt.plot(df.sales_outRem, color='red',label='out_rem')

    if save:
        plt.savefig('graphs/plot_'+str(storeId)+'_'+str(familyId) + family+'.jpg')
    else:
        plt.show(block=False)


def create_sequences(data, length):
    # Compute the strides and itemsize
    strides = (data.strides[0], data.strides[1], data.strides[0])
    shape = (data.shape[0] - length + 1, data.shape[1], length)

    # Create a view into the array with the given shape and strides
    sequences = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides, writeable=False)

    return sequences

def flattenInto2d(train, look_back, predictedVal, a, b, l):
    maxLen = train.shape[0] - look_back-predictedVal
    a1 = np.reshape(a[0:maxLen,:,:], (maxLen,-1))
    b1 = np.reshape(b[0:maxLen,:,:], (maxLen,-1))
    X = np.concatenate((a1,b1), axis=1)
    y = np.reshape(l, (maxLen,-1))

    return X,y

def getSequencesFast(train, trainF, look_back, n_predictedValues, zScoreNorm = False, applyZScoreNorm = False, meanZ = 0, stdZ = 0):
    # zscore over all values -> not ideal bc test data
    if zScoreNorm:
        #mean = train.sales.mean()
        mean = 0 # modified zScore, not in mean = 0
        std = max(train.sales.std(), 1)
        train.loc[:,'sales'] = (train.sales - mean) / std
    if applyZScoreNorm:
        train.loc[:,'sales'] = (train.sales-meanZ)/stdZ

    trainF2 = trainF + ['sales']


    pastS   = create_sequences(train[trainF2].to_numpy(), look_back)                                   #past sequence
    futureS = create_sequences(train[trainF].iloc[look_back:-1].to_numpy(), n_predictedValues)        #future sequence
    label   = create_sequences(train[['sales']].iloc[look_back:-1].to_numpy(), n_predictedValues) #label

    X, y = flattenInto2d(train, look_back, n_predictedValues, pastS, futureS, label)

    if zScoreNorm:
        return X, y, mean, std
    elif applyZScoreNorm:
        return X,y
    else:
        return X,y

def getSimpleSequence(train, n_predictedValues, features, zScoreNorm = False, applyZScoreNorm = False, meanZ = 0, stdZ = 0):
    """
    only a 1d sequence, e.g. a 16 strided sequence, the past features have to be in the train df already included
    """
    sequence0 = []
    labels = []

    # zscore over all values -> not ideal bc test data
    if zScoreNorm:
        mean = train.sales.mean()
        mean = 0 # modified zScore, not in mean = 0
        std = max(train.sales.std(), 1)
        train.loc[:,'sales'] = (train.sales - mean) / std
    if applyZScoreNorm:
        train.loc[:,'sales'] = (train.sales-meanZ)/stdZ

    for i in range(train.shape[0]-n_predictedValues):
        startS0 = i
        endS0 = startS0 + n_predictedValues
        sequence0.append(train[features].iloc[startS0:endS0].to_numpy())
        labels.append(train['sales'].iloc[startS0:endS0])

    X = np.stack(sequence0, axis = 0)
    y    = np.stack(labels, axis = 0)

    if zScoreNorm:
        return X, y, mean, std
    elif applyZScoreNorm:
        return X,y
    else:
        return X,y

def getSequencesFastNOTFlattened(train, trainF, look_back, n_predictedValues, zScoreNorm = False, applyZScoreNorm = False, meanZ = 0, stdZ = 0):
    # zscore over all values -> not ideal bc test data
    if zScoreNorm:
        #mean = train.sales.mean()
        mean = 0 # modified zScore, not in mean = 0
        std = max(train.sales.std(), 1)
        train.loc[:,'sales'] = (train.sales - mean) / std
    if applyZScoreNorm:
        train.loc[:,'sales'] = (train.sales-meanZ)/stdZ

    trainF2 = trainF + ['sales']


    pastS   = create_sequences(train[trainF2].to_numpy(), look_back)                                   #past sequence
    futureS = create_sequences(train[trainF].iloc[look_back:-1].to_numpy(), n_predictedValues)        #future sequence
    label   = create_sequences(train[['sales']].iloc[look_back:-1].to_numpy(), n_predictedValues) #label


    if zScoreNorm:
        return pastS, futureS, label, mean, std
    elif applyZScoreNorm:
        return pastS, futureS, label
    else:
        return pastS, futureS, label

def removeOutliers(a):
     # check if lots of 0s
    counts, bins = np.histogram(a.sales, bins=50)
    binZero = counts[0]
    binNextZero = -1
    for i,count in enumerate(counts):
        if i > 0 and count != 0:
            binNextZero = count
            break
    isZeroSinglePeak = binZero > 2*binNextZero
    countsSorted = np.sort(counts)[::-1]
    significantZeroPart = binZero > countsSorted[1]
    fishy = (significantZeroPart and isZeroSinglePeak)# or not isStationary


    # remove outliers  ---- seems to work ok-ish
    a.loc[:,'rolling7'] = a.sales.rolling(14).mean()
    a.loc[:,'rolling7std'] = a.sales.rolling(14).std()
    a.loc[:,'rollingThreshold'] = (a['rolling7'] + 5* a['rolling7std']).shift(1)
    a['absMean'] = a.sales.mean() + 5*a.sales.std()
    a['sales_outRem'] = a.sales
    if fishy:
        a.loc[(a.sales>2*a.absMean) & (a.sales>a.rollingThreshold) & (a.sales>20),'sales_outRem'] = np.nan
    else:
        a.loc[(a.sales>a.absMean) & (a.sales>a.rollingThreshold),'sales_outRem'] = np.nan
    hasOutliers = a.sales_outRem.isna().sum()
    a['sales_outRem'] = a.sales_outRem.interpolate(limit_direction='both')
    return a

def filterDataForOutliersDeprecated(data, familyId, storeId, flippedPropDicts, checkStationary = False, render = False, saveFig = False):
    a = data.loc[(data.store_nbr == storeId) & (data.family == familyId) & (data.dataT == 'train')].copy()

    #remove feb29
    #a = a.loc[~((a.date.dt.day==29) & (a.date.dt.month==2))]

    if 0: # do not filter away initial periods
        a.loc[:,'cumsum0'] = a.sales.cumsum()

        # filter out if product is not offered
        a = a.loc[a.cumsum0 > 0]

    # only consider data after july 2015 /other stuff seems to be too old
    a = a.loc[a.date > "2015-07-01"]

    # check if stationary
    if checkStationary:
        try:
            #p_value = test_stationarity(np.log(a.sales+1).diff(21), 12, True)
            p_value = test_stationarity(a.sales, 12, False)
        except:
            p_value = 1e6
        isStationary = p_value < 0.05
    
    # check if lots of 0s
    counts, bins = np.histogram(a.sales, bins=50)
    binZero = counts[0]
    binNextZero = -1
    for i,count in enumerate(counts):
        if i > 0 and count != 0:
            binNextZero = count
            break
    isZeroSinglePeak = binZero > 2*binNextZero

    countsSorted = np.sort(counts)[::-1]
    significantZeroPart = binZero > countsSorted[1]

    fishy = (significantZeroPart and isZeroSinglePeak)# or not isStationary

    # remove outliers  ---- seems to work ok-ish
    a.loc[:,'rolling7'] = a.sales.rolling(14).mean()
    a.loc[:,'rolling7std'] = a.sales.rolling(14).std()
    a.loc[:,'rollingThreshold'] = (a['rolling7'] + 5* a['rolling7std']).shift(1)
    a['absMean'] = a.sales.mean() + 5*a.sales.std()
    a['sales_outRem'] = a.sales

    if fishy:
        a.loc[(a.sales>2*a.absMean) & (a.sales>a.rollingThreshold) & (a.sales>20),'sales_outRem'] = np.nan
    else:
        a.loc[(a.sales>a.absMean) & (a.sales>a.rollingThreshold),'sales_outRem'] = np.nan

    hasOutliers = a.sales_outRem.isna().sum()
    a['sales_outRem'] = a.sales_outRem.interpolate(limit_direction='both')

    
    if render and (isfishy or hasOutliers):
        print(storeId, familyId, 'stationary ',isStationary, p_value, 'n_outliers: ',hasOutliers,flippedPropDicts['family'][familyId])
        plotSales(a, storeId, flippedPropDicts['family'][familyId], familyId, p_value, save=saveFig)
    return a

def getSequencesDeprecated(train, trainF, trainF2, look_back, n_predictedValues, zScoreNorm = False, applyZScoreNorm = False, meanZ = 0, stdZ = 0):
    sequence0 = []
    sequence1 = []
    labels = []

    # zscore over all values -> not ideal bc test data
    if zScoreNorm:
        mean = train.sales.mean()
        mean = 0 # modified zScore, not in mean = 0
        std = max(train.sales.std(), 1)
        train.loc[:,'sales'] = (train.sales - mean) / std
    if applyZScoreNorm:
        train.loc[:,'sales'] = (train.sales-meanZ)/stdZ
    for i in range(train.shape[0]-look_back-n_predictedValues):
        startS0 = i
        endS0 = startS0 + look_back
        endS1 = endS0 + n_predictedValues
        sequence0.append(train[trainF2].iloc[startS0:endS0].to_numpy().flatten())
        sequence1.append(train[trainF].iloc[endS0:endS1].to_numpy().flatten())
        labels.append(train['sales'].iloc[endS0:endS1])

    sequence0 = np.stack(sequence0, axis = 0)
    sequence1 = np.stack(sequence1, axis=0)
    labels    = np.stack(labels, axis = 0)


    X = np.concatenate((sequence0, sequence1), axis=1)
    y = labels

    if zScoreNorm:
        return X, y, mean, std
    elif applyZScoreNorm:
        return X,y
    else:
        return X,y

def sanityChecks(holidays, stores, train4, train5):
    # check that the city holidays are the same in the holiday and store df
    uniqueLocalsHolidays = holidays.loc[holidays.locale =='Local'].locale_name.unique()
    uniqueLocalCities = stores.city.unique()

    intersection = set(uniqueLocalsHolidays).intersection(set(uniqueLocalCities))
    not_intersection_list1 = set(uniqueLocalsHolidays).difference(intersection)
    not_intersection_list2 = set(uniqueLocalCities).difference(intersection)

    print(intersection)
    print(not_intersection_list1)
    print(not_intersection_list2)
    #result: we have a couple cities without holidays, but that is fine, the rest is the same 

    # Sanity check, somewhat ok
    rows = train4.shape[0]
    holTypes = rows - train4.holidayType.isna().sum()
    holTypes1 = rows - train4.type_reg.isna().sum()
    holTypes2 = rows - train4.type_nat.isna().sum()
    sumTypes = holTypes + holTypes1 + holTypes2
    print(sumTypes, rows - train5.holidayType.isna().sum())

    holTypes = rows - train4.description.isna().sum()
    holTypes1 = rows - train4.description_reg.isna().sum()
    holTypes2 = rows - train4.description_nat.isna().sum()
    sumTypes = holTypes + holTypes1 + holTypes2
    print(sumTypes, rows - train5.description.isna().sum())

    holTypes = rows - train4.transferred.isna().sum()
    holTypes1 = rows - train4.transferred_reg.isna().sum()
    holTypes2 = rows - train4.transferred_nat.isna().sum()
    sumTypes = holTypes + holTypes1 + holTypes2
    print(sumTypes, rows - train5.transferred.isna().sum())


def addLaggedFutureHolidays(storeDf, features = ['transferred', 'holidayType'], lags = 10):
       transferredF = []
       for i in range(lags):
              lag = i+1 # (1-5)
              for f in features:
                     f0 = f+'_lag'+str(lag)
                     f1 = f+'_lag-'+str(lag)
                     storeDf.loc[:,[f0]] = storeDf[f].shift(lag).fillna(0)
                     storeDf.loc[:,[f1]] = storeDf[f].shift(-lag).fillna(0)
                     transferredF.append(f0)
                     transferredF.append(f1)
       return storeDf, transferredF


def dataProcessing(storeDf, date_string_val, date_string_test, predictDiff, seasonalFDiff, seasonalLags, targetLags, rolling, initial_lag, prepPred = False):
       """ create training data based on lagged features not 2 sequences """
       trainF = [
              #'store_nbr', 'family', 
              #'sales', 
              # 'dataT',
              #'city', 'state', 'type', 'cluster', 
              #'dcoilwtico','onpromotion',  # added down the line!
              'holidayType',
              'description', 
              'transferred', 
              #'transactions', 
              'store_closed',
              'weekday_0', 'weekday_1', 'weekday_2',
              'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6',
              #'type_0','type_1','type_2','type_3','type_4', #store type
              'holidayType_0','holidayType_1','holidayType_2','holidayType_3','holidayType_4','holidayType_5','holidayType_6',
              'description_0','description_1','description_2','description_3','description_4','description_5','description_6','description_7','description_8','description_9','description_10','description_11','description_12','description_13','description_14','description_15','description_16','description_17','description_18'
   
              ]
       timeF = [
              'linear_time', 'day_of_year', 'day_of_year_f12_0', 'day_of_year_f104_0','day_of_year_f24_0',  'day_of_year_f52_0',
              'day_of_year_f12_180', 'day_of_year_f104_180','day_of_year_f24_180','day_of_year_f52_180', 
              #'weekday', 
              'month'
              ]
       if not prepPred:
              storeDf = storeDf.loc[(storeDf.dataT == 'train')]
              mask = (storeDf.date >= date_string_val)
              storeDf.loc[mask,['dataT']] = 'val'

       # ln tranformation
       storeDf.loc[:,['logSales']] = np.log(storeDf.sales + 1)
       storeDf.loc[:,['transactions']] = storeDf['transactions'].fillna(0)
       storeDf['onpromotion'] = (storeDf['onpromotion']- storeDf.onpromotion.mean()) /storeDf.onpromotion.std() # helps against overfitting  # 

       
       relevantSales = storeDf.loc[storeDf.dataT =='train']
       dfLen = storeDf.shape[0]


       seasonF = []
       for period in [7]: #, 14,21,28,52  104, 365
              for pf in ['logSales','transactions','dcoilwtico']:
                     dec = sm.tsa.seasonal_decompose(relevantSales[pf],period = period, model = 'additive')
                     #print(period, max(dec.seasonal))
                     f = pf+'Seasonality_'+str(period)
                     storeDf.loc[:,[f]] = addSeasonality(period, dec, dfLen)
                     seasonF.append(f)

       seasonF.append('dcoilwtico')
       seasonF.append('onpromotion')   # helps against overfitting
       #print('done with seasonal decompose')

       if predictDiff:
              storeDf.loc[:,['ref']] = storeDf['logSales'].shift(initial_lag)
              storeDf.loc[:,['target']] = storeDf['logSales'] - storeDf['ref']
       else:
              storeDf.loc[:,['target']] = storeDf['logSales']

              
       seasonDiffF2 = []
       for f in seasonF:
              newF = f+'_diff'+str(seasonalFDiff)
              storeDf.loc[:,[newF]] = storeDf[f].diff(seasonalFDiff)
              seasonDiffF2.append(newF)
       
       # seasonal featuers lags
       seasonalFLags = seasonF + seasonDiffF2
       featuresForSLag = []#trainF
       for i in seasonalLags:
              lag = i
              newF = [seasonalFLags[j] + '_lag' + str(lag) for j in range(len(seasonalFLags))]
              featuresForSLag = featuresForSLag + newF
              storeDf.loc[:,newF] = storeDf[seasonalFLags].shift(lag).to_numpy()

       seasonalFeatures = seasonF + seasonDiffF2 + featuresForSLag
       #print('done with seasonal features')
       

       
       # lag features / how many past datapoints are we tain
       featuresForLag = ['target']
       targetLagF = []#trainF
       for i in targetLags:
              lag = i+initial_lag
              newF = [featuresForLag[j] + '_lag' + str(lag) for j in range(len(featuresForLag))]
              targetLagF = targetLagF + newF
              storeDf.loc[:,newF] = storeDf[featuresForLag].shift(lag).to_numpy()
       #print('done with target lags')
       
       #------ also add future holidays!--------
       storeDf, transferredF = addLaggedFutureHolidays(storeDf, features=['transferred','holidayType'], lags=10)

       # rolling features
       createdF = seasonF + targetLagF
       rollingF = []
       for rol in rolling:
              for i in range(len(createdF)):
                     #if 'sales_t-16'  in lagF[i]:
                     #if createdF[i] in ['target','dcoilwtico']:# or 'dcoilwtico' in lagF[i]:#'target'  in lagF[i]:
                            fm = createdF[i]+'_rollingM' + str(rol)
                            fs = createdF[i]+'_rollingS' + str(rol)
                            rollingF.append(fm)
                            rollingF.append(fs)
                            storeDf.loc[:,[fm]] = storeDf[createdF[i]].rolling(rol).mean()#.copy()
                            storeDf.loc[:,[fs]] = storeDf[createdF[i]].rolling(rol).std()#.copy()
              #print('done with rolling:', rol)
       #print('done with rolling')


       allF = rollingF + timeF + trainF +transferredF +seasonalFeatures + targetLagF
       # we get a matrix that predicts only 1 timestamp -> stride it
       if len(rolling) == 0:
              storeDf = storeDf.iloc[max(max(targetLags), max(seasonalLags))+initial_lag+1:storeDf.shape[0]]
       else:
              storeDf = storeDf.iloc[max(max(targetLags), max(seasonalLags))+initial_lag+max(rolling)+1:storeDf.shape[0]]

       
       if prepPred:
              train_subDf = storeDf.loc[(storeDf.date < date_string_test) & (storeDf.dataT == 'train')]
              test_subDf  = storeDf.loc[(storeDf.date >= date_string_test)& (storeDf.dataT == 'train')]
              pred_subDf = storeDf.loc[storeDf.dataT == 'test']
       else:
              train_subDf = storeDf.loc[storeDf.date < date_string_test]
              test_subDf  = storeDf.loc[storeDf.date >= date_string_test]
              pred_subDf = storeDf.loc[storeDf.dataT =='val']
    
       return train_subDf, test_subDf, pred_subDf, allF


def processAllData(data1, targetLags, featureLags, rolling, initial_lag, date_string_val=None):
    grouped = data1.groupby(['store_nbr','family'])
    data1['transactions'] = (data1.transactions - grouped.transactions.transform('mean')) / grouped.transactions.transform('std')
    data1['linear_time'] = (data1['linear_time'] - grouped.linear_time.transform('mean')) / grouped.linear_time.transform('std')
    data1['day_of_year'] = (data1['day_of_year'] - grouped.day_of_year.transform('mean')) / grouped.day_of_year.transform('std')
    data1['dcoilwtico'] = (data1['dcoilwtico'] - grouped.dcoilwtico.transform('mean')) / grouped.dcoilwtico.transform('std')   

    if date_string_val != None:
        mask = (data1.date >= date_string_val) & (data1.dataT == 'train')
        data1.loc[mask,['dataT']] = 'val'  


    data1.loc[:,['logSales']] = np.log(data1.sales + 1)
    arimaPred = pd.read_csv('csvs\sarima_313_117_and_id.csv')
    data1 = pd.merge(data1, arimaPred[['id','sales','salesArima']], on=['id','sales'], how='left')
    print('loaded arima data') 
    data1['ref'] = data1['salesArima']
    data1['target'] = data1['logSales'] - data1['ref']
    
    featuresForLag = ['target']
    lagF_target = []
    for l in targetLags:
           lag = l + initial_lag
           newF = [featuresForLag[j] + '_lag' + str(lag) for j in range(len(featuresForLag))]
           lagF_target = lagF_target + newF
           data1[newF] = data1.groupby(['store_nbr','family'])[featuresForLag].shift(lag)  



    featuresForLag2 = ['salesArima','onpromotion','dcoilwtico']
    lagF_features = []
    for i in featureLags:
           lag = i
           newF = [featuresForLag2[j] + '_lag' + str(lag) for j in range(len(featuresForLag2))]
           lagF_features = lagF_features + newF
           data1[newF] = data1.groupby(['store_nbr','family'])[featuresForLag2].shift(lag) 

    

    lagF = lagF_target + lagF_features #+ groupedF
    rollingF = []
    for rol in rolling:
           for i in range(len(lagF)):
                  #if 'sales_t-16'  in lagF[i]:
                  if 'target'  in lagF[i] or 'dcoilwtico'  in lagF[i] or 'onpromotion'  in lagF[i] or 'salesArima'  in lagF[i]:
                         fm = lagF[i]+'_rollingM' + str(rol)
                         fs = lagF[i]+'_rollingS' + str(rol)
                         rollingF.append(fm)
                         rollingF.append(fs)
                         data1[fm] = data1.groupby(['store_nbr','family'])[lagF[i]].rolling(rol, min_periods=1).mean().reset_index(drop=True)#.set_index('id')#.reset_index() #transform('mean') #lambda x: x.rolling(rol).mean()).to_numpy()
                         data1[fs] = data1.groupby(['store_nbr','family'])[lagF[i]].rolling(rol, min_periods=1).std().reset_index(drop=True)  

    groupedF = []
    featuresForGroups = ['onpromotion','salesArima'] #,'transactions'
    for gFeature in ['family','cluster','type','store_nbr','city','state']:
        newFGrouped = [featuresForGroups[j] + '_per' + gFeature for j in range(len(featuresForGroups))]
        data1[newFGrouped] = data1.groupby(['date',gFeature])[featuresForGroups].transform('sum')
        groupedF = groupedF + newFGrouped

    featuresForLag3 = groupedF
    lagF_features1 = []
    for i in featureLags:
           lag = i
           newF = [featuresForLag3[j] + '_lag' + str(lag) for j in range(len(featuresForLag3))]
           lagF_features1 = lagF_features1 + newF
           data1[newF] = data1.groupby(['store_nbr','family'])[featuresForLag3].shift(lag) 
    

    data3 = data1.groupby(['store_nbr','family']).apply(lambda x: x.iloc[max(max(targetLags),max(featureLags))+initial_lag+1:])
    data3 = data3.set_index('id').reset_index()
    return data3, lagF + rollingF +lagF_features1+groupedF
