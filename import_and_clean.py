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


def subscriptions():
    '''Imports, cleans, and returns subscriptions.csv as pandas DataFrame.

    This function imports and cleans subscriptions.csv. There are a lot of repeated account.id's in this
    file so I aggregated a lot of statistics about them as I saw fit so that all data was contained in a 
    single row for each account.id --> No account.id repeats! This also performs onehot encoding on 
    the "package" and "section" categorical features. 

    Args:
        (None)

    Returns:
        (Pandas DataFrame): cleaned subscriptions.csv
    '''
    subscriptions = pd.read_csv('./data/subscriptions.csv')

    package_and_section = subscriptions.copy() # use this to one hot encode package and section values
    package_and_section = package_and_section.drop(labels = ['season','no.seats','location','price.level','subscription_tier','multiple.subs'], axis =1) 
    package_and_section['section'] = package_and_section['section'].fillna(package_and_section['section'].mode()[0])
    package_and_section['package'] = package_and_section['package'].fillna(package_and_section['package'].mode()[0])

    # one hot encoding of "package" and "section" categorical features
    onehot_enc = OneHotEncoder(handle_unknown= 'ignore')
    cols = ['package','section']
    onehot_enc.fit(package_and_section[cols])
    colnames = list(onehot_enc.get_feature_names(input_features=cols))
    onehot_vals = onehot_enc.transform(package_and_section[cols]).toarray()
    enc_df = pd.DataFrame(onehot_vals, columns = colnames, index = package_and_section['account.id'])
    package_and_section = pd.concat([package_and_section,enc_df.reset_index(drop = True)], axis = 1)
    package_and_section = package_and_section.drop(labels = cols, axis = 1)
    package_and_section = package_and_section.groupby(package_and_section['account.id']).aggregate('sum')
    package_and_section = package_and_section.reset_index()


    subscriptions = subscriptions.set_index('account.id')
    subscriptions = subscriptions.drop(labels = ['location','package','section'], axis = 1)
    subscriptions['multiple.subs'] = subscriptions['multiple.subs'].map({'yes': 1, 'no': 0})
    subscriptions['season'] = subscriptions['season'].str.split('-').str[0].astype(int)
    subscriptions['most_recent_season'] = subscriptions.groupby('account.id')['season'].max() # most_recent_season
    subscriptions = subscriptions.drop('season', axis = 1)
    subscriptions['total_num_seats'] = subscriptions.groupby('account.id')['no.seats'].sum() # most_recent_season
    subscriptions = subscriptions.drop('no.seats', axis = 1)
    subscriptions['avg_sub_price_level'] = subscriptions.groupby('account.id')['price.level'].mean() # most_recent_season
    subscriptions = subscriptions.drop('price.level', axis = 1)
    subscriptions['avg_sub_tier'] = subscriptions.groupby('account.id')['subscription_tier'].mean() # most_recent_season
    subscriptions = subscriptions.drop('subscription_tier', axis = 1)
    subscriptions['total_num_subs'] = subscriptions.groupby('account.id')['multiple.subs'].count() # most_recent_season
    subscriptions = subscriptions.drop('multiple.subs', axis = 1)
    subscriptions = subscriptions.reset_index()
    subscriptions = subscriptions.drop_duplicates(subset = 'account.id')

    subscriptions = subscriptions.merge(package_and_section, how = 'inner', on = 'account.id')

    return subscriptions


def tickets_all():
    '''Imports, cleans, and returns tickets_all.csv as pandas DataFrame.

    This function imports and cleans tickets_all.csv. It aggregates the data for each account.id 
    so that each row is a unique account.id with aggregate data describing the user's ticket
    purchasing history as follows:

    total_num_ticket_purchases_historical --> total number of times this account has purchased tickets 
    total_num_ticket_seats --> total number of ticket seats that have been purchased 
    most_recent_ticket_season --> most recent season a ticket was purchased
    avg_ticket_price_level --> average price level of tickets in purchase history

    Args:
        (None)
    
    Returns:
        (Pandas DataFrame): cleaned tickets_all.csv
    '''
    tickets_all = pd.read_csv('./data/tickets_all.csv')
    tickets_all = tickets_all.drop(labels = ['marketing.source','location','set','multiple.tickets'], axis = 1)

    tickets_all['total_num_ticket_purchases_historical'] = tickets_all.groupby('account.id')['season'].transform('count')
    tickets_all['total_num_ticket_seats'] = tickets_all.groupby('account.id')['no.seats'].transform('sum')

    tickets_all['season'] = tickets_all['season'].str.split('-').str[0]
    tickets_all['most_recent_ticket_season'] = tickets_all.groupby('account.id')['season'].transform('max')

    tickets_all['price.level'] = pd.to_numeric(tickets_all['price.level'], errors = 'coerce')
    tickets_all['avg_ticket_price_level'] = tickets_all.groupby('account.id')['price.level'].transform('mean')

    tickets_all = tickets_all.drop(labels = ['no.seats','season','price.level'], axis = 1)
    tickets_all = tickets_all.drop_duplicates(subset = 'account.id')

    return tickets_all


def train():
    '''Imports and returns account id's with associated training labels.
    
    Args:
        (None)

    Returns:
        (Pandas DataFrame): train.csv
    '''
    train = pd.read_csv('./data/train.csv')
    return train