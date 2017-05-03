import numpy as np
import matplotlib.pyplot as pt

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

from process_data import *

# Like decision trees, forests of trees also extend to multi-output problems (if Y is an array of size [n_samples, n_outputs]).

pt.ion()


def preprocess_feature(train, test):
    # ----- ------------------------------ normalization
    cols_to_norm = ['Credit amount', 'Age','Duration']
    train, test = normalize_feature(train, test, cols_to_norm)

    train.head()

    return train, test

#visual(train)

# Converting a categorical feature
# combine = [train, test]



def setup_expand_all(train, test):
    '''

    :param train:
    :param test:
    :return: train, test
    '''

    return train, test

def setup_unorder_all(train, test):
    '''
    turn numerical distribution into categories
    unordered categories for all
    :param train:
    :param test:
    :return:
    '''

    full_data = [train, test]

    # for data in full_data:
    #     data['Sex'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int)

    for data in full_data:
        data['Credit amount'] = pd.cut(data['Credit amount'], bins= 3, labels=['little', 'moderate', 'rich'])
        data['Age'] = pd.cut(data['Age'], bins=3, labels=['young', 'middle', 'old'])
        data['Duration'] = pd.cut(data['Duration'], bins=3, labels=['short', 'middle', 'long'])

    # categorized data, with the original categorized data
    cat_vars = list(train.keys())
    cat_vars.remove('Class')
    # cat_vars.remove('Sex')
    # cat_vars = ['Credit amount','Housing', 'Age', 'Duration', 'Checking account', 'Saving accounts', 'Job', 'Purpose']
    train = pd.get_dummies(train, columns=cat_vars)
    test = pd.get_dummies(test, columns=cat_vars)
    # mapping

    return train, test

def setup_order_all(train, test, map = 1):
    '''
    put all to be continuous, 
    yet the feature mapping is done deliberately considering the initial observation
    :param train: 
    :param test: 
    :return: 
    '''
    full_data = [train, test]

    if map == 1:
        acc_mapping = {"missing": 0, "little": 1, "moderate": 2, "quite rich": 3, "rich": 4}
    else:
        acc_mapping = {"missing": 0, "little": 1, "moderate": 1, "quite rich": 2, "rich": 2}


    for data in full_data:
        data['Credit amount'] = pd.cut(data['Credit amount'], bins= 3, labels=['little', 'moderate', 'rich'])
        data['Sex'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int)
        data['Checking account'] = data['Checking account'].map(acc_mapping)
        data['Saving accounts'] = data['Saving accounts'].map(acc_mapping)
        data['Credit amount'] = data['Credit amount'].map(acc_mapping)


    return train, test
        
    
def naive_bayes_features(train, test):
    train, test = drop_select(train, test, ['Purpose'])
    train, test = setup_unorder_all(train, test)
    # cat_vars = ['Housing']

    # train = pd.get_dummies(train, columns=cat_vars)
    # test = pd.get_dummies(test, columns=cat_vars)


    return train, test


def drop_select(train, test, columns):

    train = train.drop(columns, axis=1)
    test = test.drop(columns, axis=1)

    return train, test



# --- --- --- --- --- --- --- from categorical to numerical
# cat_vars=['Purpose', 'Checking account', 'Saving accounts', 'Housing']
# second choice of mapping
# combine rich and quite rich for saving
# train['Checking account'] = train['Checking account'].map({"quite rich": 'rich'})
# train['Saving accounts'] = train['Saving accounts'].map({"moderate": 'little'})

# ---- ---- ----- ---- ---- drop columns
#
# def user_defined(train, test):

#
#
#     # ----- -- ---- --- ---- from numerical/string to categorical
#
#     cat_vars =[]
#
#     cat_vars = ['Age']
#
#     # ---- ----- ---- ---- ---- map variables
#
#         # data = data.drop('Checking account_missing', axis=1)
#         # data = data.drop('Saving accounts_missing', axis=1)
#
        # data['Housing'] = data['Housing'].map({'free': 0, 'rent': 1, 'own': 2})

#
