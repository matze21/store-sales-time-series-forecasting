import pandas as pd
from baseFunctions import *

def processData6():
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

    data5['description'] = data5['description'].combine_first(data5['description_reg'])
    data5['description'] = data5['description'].combine_first(data5['description_nat']) 

    data5['transferred'] = data5['transferred'].combine_first(data5['transferred_reg'])
    data5['transferred'] = data5['transferred'].combine_first(data5['transferred_nat']) 

    data5 = data5.drop(columns=['type_reg','type_nat','description_reg','description_nat','transferred_reg','transferred_nat']) 

    print(data4.shape, data5.shape) 
    print('finished data5')
    data6 = data5.copy()
    propDicts = {}
    for f in ['family','city','state','type','holidayType','description','transferred']:
        unique = data6[f].unique()
        category_dict = {category: index for index, category in enumerate(unique)}
        data6[f] = data6[f].map(category_dict)
        propDicts[f] = category_dict    

    flippedPropDicts = {}
    for key,value in propDicts.items():
        flippedPropDicts[key] = {value: key for key, value in propDicts[key].items()}

    return data6, propDicts, flippedPropDicts

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
