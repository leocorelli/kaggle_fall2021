import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def accounts():
    '''Imports, cleans, and returns account.csv as pandas DataFrame.

    This function imports account.csv and removes features: ['shipping.zip.code', 'shipping.city','relationship','first.donated']. 
    It converts all 'billing.city' to lowercase to make merging easier. It then casts zipcodes as string objects since some 
    zipcodes have non-numeric characters, making merging possible. 

    Args:
        (None)
    
    Returns:
        (Pandas DataFrame): cleaned account.csv
    '''
    accounts = pd.read_csv('./data/account.csv',encoding='latin-1')
    accounts = accounts.drop(labels = ['shipping.zip.code', 'shipping.city','relationship','first.donated'], axis = 1)
    accounts['billing.city'] = accounts['billing.city'].str.lower()
    accounts['billing.zip.code'] = accounts['billing.zip.code'].astype(str)

    return accounts

def zipcodes():
    '''Imports, cleans, and returns zipcodes.csv as pandas DataFrame.
        
    This function imports zipcodes.csv. It keeps columns: ['Zipcode','City','Lat','Long','TaxReturnsFiled','EstimatedPopulation','TotalWages']. 
    It then casts 'city' as a string and makes it lowercase, making merging easier. It then casts zipcodes as strings since there are non-numeric 
    characters, also making merging easier. 

    Args:
        (None)

    Returns:
        (Pandas DataFrame): cleaned zipcodes.csv
    '''
    zipcodes = pd.read_csv('./data/zipcodes.csv')
    zipcodes = zipcodes[['Zipcode','City','Lat','Long','TaxReturnsFiled','EstimatedPopulation','TotalWages']]
    zipcodes['City'] = zipcodes['City'].astype(str)
    zipcodes['City'] = zipcodes['City'].str.lower()
    zipcodes['Zipcode'] = zipcodes['Zipcode'].astype(str)

    return zipcodes






