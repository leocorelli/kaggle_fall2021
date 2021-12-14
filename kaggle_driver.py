from import_and_clean import accounts, zipcodes, subscriptions, tickets_all, train
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

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


def prepare_for_modeling(X, y):
    '''Splits labeled data into training and test sets. Then, scales data and returns.

    Args:
        X (Pandas DataFrame): training data
        y (Pandas DataFrame): training labels
    
    Returns: 
        X_train (Numpy ndarray): scaled training data to train model
        X_test (Numpy ndarray): scaled test data to generate predictions on
        y_train (Numpy ndarray): training target labels
        y_test (Numpy ndarray): test labels to compare predictions to
    '''
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 0, test_size = 0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def train_and_predict(X_train, X_test, y_train, y_test):
    model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.6392392088970968, gamma=0,
              learning_rate=0.14617846831708017, max_delta_step=0, max_depth=1,
              min_child_weight=3, missing=None, n_estimators=89, n_jobs=1,
              nthread=None, eval_metric='logloss', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=0.8490861621273507, verbosity=1, use_label_encoder = False)

    model.fit(X_train,y_train)
    prob_preds = model.predict_proba(X_test)[:,-1]
    auroc = roc_auc_score(y_test,prob_preds)

    return auroc


if __name__ == '__main__':
    X, y = get_data()
    X_train, X_test, y_train, y_test = prepare_for_modeling(X,y)
    auroc = train_and_predict(X_train, X_test, y_train, y_test)
    print(auroc)
    