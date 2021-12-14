from import_and_clean import accounts, zipcodes, subscriptions, tickets_all, train
import pandas as pd

def get_data():
    '''Imports all data files and returns X (all merged data) and y (training labels).

    Args:
        (None)

    Returns:
        X (Pandas DataFrame): training data
        y (Pandas Series): training labels
    '''
    acc = accounts()
    zip = zipcodes()
    sub = subscriptions()
    tix = tickets_all()
    X = train()

    data = pd.merge(acc,zip, how = 'left', left_on = 'billing.zip.code', right_on = 'Zipcode').drop(labels = ['Zipcode','City'], axis = 1)
    data = data.merge(sub, how = 'left', on = 'account.id')
    data = data.merge(tix, how = 'left', on = 'account.id')
    data = data.drop(labels = ['billing.zip.code','billing.city'], axis = 1) 

    data = data.merge(X, how = 'right', on = 'account.id') # all labeled data that I have available

    # begin final data preparation 

    y = data['label'] # targets
    X = data.drop(labels = ['label','account.id'], axis = 1) # remove targets from X
    

    return X, y



if __name__ == '__main__':
    X, y = get_data()
    print(y)