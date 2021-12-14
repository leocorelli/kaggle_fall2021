import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def accounts():
    '''
        This function imports account.csv, drops unnecessary features, clenas, and returns DataFrame.

        This function imports account.csv and removes unnecessary features. 
        It then does some basic cleaning (casting zipcodes as strings since some zipcodes have 
        non-numeric characters) and returns the final dataframe for merging.


        Args:
            None
        
        Returns:
            Pandas DataFrame: cleaned account.csv

    '''
    accounts = pd.read_csv('./data/account.csv',encoding='latin-1')
    accounts = accounts.drop(labels = ['shipping.zip.code', 'shipping.city','relationship','first.donated'], axis = 1)
    accounts['billing.city'] = accounts['billing.city'].str.lower()
    accounts['billing.zip.code'] = accounts['billing.zip.code'].astype(str)

    return accounts