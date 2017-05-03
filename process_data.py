
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import stats
import random


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))


def is_valid_samples(samples):
    '''
    from a array of sample, get invalid, aka NA entry
    :param samples: array
    :return: boolean array
    '''
    # check whether is float object
    checkSample = [isinstance(i, float) for i in samples]
    return  ~np.asarray(checkSample)


def get_mode(features, target):
    validSamples = is_valid_samples(features[target])
    validFeatures = features[validSamples == True]
    sampleMode = stats.mode(validFeatures[target])

    return sampleMode[0]


def validate_samples(features, target, action = 'delete'):
    '''

    :param features:
    :param target:
    :param validSamples:pwd
    :param action: {'delete', 'mode', 'mean'}
    :return:
    '''

    validSamples = is_valid_samples(features[target])

    if action == 'delete':
        features= features[validSamples == True]

    # fill invalid with mean - note: must be number
    elif action == 'mean':
        validFeatures = features[validSamples == True]
        sampleMean = np.mean(validFeatures[target])
        updateFeatures = [sampleMean if isinstance(i, float) else i for i in features[target]]
        features[target] = np.asarray(updateFeatures)

    elif action == 'mode':
        validFeatures = features[validSamples == True]
        sampleMode = stats.mode(validFeatures[target])
        updateFeatures = [sampleMode[0] if isinstance(i, float) else i for i in features[target]]
        features[target] = np.asarray(updateFeatures)

    return features

def normalize_feature(train, test, columns):
    '''

    :param train:
    :param test:
    :param columns: list of columns
    :return: array of normalizaed train, test
    '''

    # train[columns] = train[columns].apply(lambda x: (x - x.mean()) * 10 / (x.max() - x.min()))
    # test[columns] = test[columns].apply(lambda x: (x - x.mean()) * 10 / (x.max() - x.min()))

    # noremalize
    # cols_to_norm = ['Credit amount', 'Duration']
    # train[columns] = train[columns].apply(lambda x: (x - x.mean()) * 10 / (x.max() - x.min()))
    # test[columns] = train[columns].apply(lambda x: np.log(x+1))

    for col in columns:
        meanTrain = train[col].mean()
        varTrain = train[col].var()
        # train[col] = (train[col] - meanTrain)/varTrain
        # test[col] = (test[col] - meanTrain) / varTrain
        train[col] = (train[col]- meanTrain) * 10 / (train[col].max() - train[col].min())
        test[col] = (test[col] - meanTrain) * 10 / (train[col].max() - train[col].min())

    return train, test

def handle_missing_data(train, test, method = 'missing'):
    '''

    :param train:
    :param test:
    :param method: {missing, mode, }
    :return: train, test

    mode: fillna with Mode of training set, since it will be unknown for test set
    '''
    if method == 'missing':
        train['Saving accounts'].fillna('missing', inplace=True)
        train['Checking account'].fillna('missing', inplace=True)

        test['Saving accounts'].fillna('missing', inplace=True)
        test['Checking account'].fillna('missing', inplace=True)

    elif method == 'mode':
        freq = train['Checking account'].dropna().mode()[0]
        train['Checking account'].fillna(freq, inplace=True)
    #     checcking account will be filled randomly
        freq = train['Saving accounts'].dropna().mode()[0]
        test['Saving accounts'].fillna(freq, inplace=True)

        train['Checking account'].fillna(freq, inplace=True)
        #     checcking account will be filled randomly
        test['Saving accounts'].fillna(freq, inplace=True)

    # todo: fix the random selection
    elif method == 'random':
        # df['Checking account'] = df['Checking account'].apply(lambda x: x.fillna(random.choice(x.dropna()), inplace=True))
        # unique = pd.Series(df['Checking account']).unique()
        # randomIn = random.choice(df['Checking account'].dropna())
        train['Saving accounts'].fillna(random.choice(train['Saving accounts'].dropna()), inplace=True)
        train['Checking account'].fillna(random.choice(train['Checking account'].dropna()), inplace=True)

    return train, test


def setup_train_test(filename):
    '''
    df = pd.read_csv('input/german_credit_data.csv')
    :param filename:
    :return:trian, test
    '''

    from sklearn.utils import shuffle


    df = pd.read_csv(filename)
    # print(df.head(5))
    df = shuffle(df)

    nmale = df[df['Sex'] == 'male'].shape[0]
    nfmale = df[df['Sex'] == 'female'].shape[0]

    print('Number of male entris:   ', nmale, ' in percent: ', float(nmale) / df.shape[0])
    print('Number of female entris: ', nfmale, ' in percent: ', float(nfmale) / df.shape[0])

    df.info()
    # drop unnecessary columns, these columns won't be useful in analysis and prediction
    # too many missing valuds for checking account
    df = df.drop(df.keys()[0], axis=1)

    # missing data in Checking account


    # 20% for testing
    train = df.sample(frac = 0.8, random_state = 42)
    test = df.drop(train.index)

    return  train, test


def setup_feature_label(train, test):

    y_test = test['Class']
    y_train = train['Class']

    X_train = train.drop('Class', axis=1)
    X_test = test.drop('Class', axis=1)

    #
    # # missing data for saving accounts
    # validIndex = is_valid_samples(X_train['Saving accounts'])
    # # get mode label
    # modeLabel = stats.mode(X_train['Saving accounts'][validIndex == True])[0]
    # X_train['Saving accounts'].fillna(modeLabel[0], inplace=True)

    return X_train, X_test, y_train, y_test




