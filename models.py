# ---- ----- Basic
import pandas as pd
import numpy as np

# ---- ----- ML models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

# ---- ----- Evaluation
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score


# ---- ----- data
import seaborn as sns


# def pca_logistic(X_train, y_train, X_test, y_test):
#     pca_data = PCA()
#     return pca_data.fit(X_train)

def get_accuracy(true, predict):

    return (sum(true == predict))*1.0/len(true)


def cross_validation_model(model, data, target, cv = 10, scoring = 'accuracy'):
    '''
    cv model, print mean, std, and f1 score
    :param model:
    :param data:
    :param target:
    :param cv:
    :param scoring:
    :return: mean, std
    '''
    crossFold = StratifiedKFold(n_splits=cv, random_state=None, shuffle=True)
    scores = cross_val_score(model, data, target, cv = crossFold, scoring=scoring)
    # mean = np.mean(scores)
    # std = np.std(scores)
    print ("cv = %d, accuracy = %.5f +/- %.5f" %(cv, scores.mean(), scores.std() ))

    return  scores.mean(), scores.std()



def cross_validation_rf(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    Y_pred = rf.predict(X_test)
    print ("accuracy is: %0.5f" %get_accuracy(y_test, Y_pred))
    print classification_report(y_test, Y_pred)

    parameters = {'n_estimators': np.linspace(10, 25, 16).astype(int)}
    n_splits = 5
    # cv = StratifiedKFold(n_splits=n_splits, random_state=None, shuffle=True)

    clf = GridSearchCV(rf, parameters, cv = 3, scoring='accuracy')
    clf.fit(X_train, y_train)
    sorted(clf.cv_results_.keys())
    mean_score = clf.cv_results_['mean_test_score']
    best_params = clf.cv_results_['params'][np.argmax(mean_score)]

    rf_new = RandomForestClassifier(n_estimators = best_params['n_estimators'])

    Y_pred = rf_new.fit(X_train, y_train).predict(X_test)
    # acc_svc = round(svc.score(X_train, y_test) * 100, 2)
    print "*** cross validation random forest clustering ***"
    print best_params['n_estimators']
    print ("accuracy is: %0.5f" %get_accuracy(y_test, Y_pred))
    print "Grid score function: ", clf.score(X_test, y_test)


    print classification_report(y_test, Y_pred)



def perceptron_cluster(X_train, y_train, X_test, y_test):
    perceptron = Perceptron()
    perceptron.fit(X_train, y_train)
    Y_pred = perceptron.predict(X_test)
    print classification_report(y_test, Y_pred)


def naive_bayesian_cluster(X_train, y_train, X_test, y_test):
    bayesian = GaussianNB()
    bayesian.fit(X_train, y_train)
    Y_pred = bayesian.predict(X_test)

    # acc_svc = round(svc.score(X_train, y_test) * 100, 2)
    print "*** naive bayesian clustering ***"
    print ("accuracy is: %0.5f" %get_accuracy(y_test, Y_pred))
    print classification_report(y_test, Y_pred)

    data = pd.concat([X_train, X_test])
    labels = pd.concat([y_train, y_test])

    # cv
    mean, std = cross_validation_model(bayesian, data, labels)
    return mean, std


def naive_bayesian_bern_cluster(X_train, y_train, X_test, y_test):
    bayesian = BernoulliNB()
    bayesian.fit(X_train, y_train)
    Y_pred = bayesian.predict(X_test)
    # acc_svc = round(svc.score(X_train, y_test) * 100, 2)
    print "*** naive bayesian with bernoulli prior clustering ***"
    print ("accuracy is: %0.5f" %get_accuracy(y_test, Y_pred))
    print classification_report(y_test, Y_pred)


def knn_cluster(X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=27)
    knn.fit(X_train, y_train)
    Y_pred = knn.predict(X_test)

    print "***knn  ***"
    print ("accuracy is: %0.5f" % get_accuracy(y_test, Y_pred))




def cross_validation_knn(X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    Y_pred = knn.predict(X_test)
    print ("accuracy is: %0.5f" %get_accuracy(y_test, Y_pred))
    print classification_report(y_test, Y_pred)

    parameters = {'n_neighbors':np.linspace(10, 40, 31)}
    n_splits = 5

    cv = StratifiedKFold(n_splits=n_splits, random_state=None, shuffle=True)
    clf = GridSearchCV(knn, parameters, cv = cv, scoring='accuracy')

    clf.fit(X_train, y_train)
    print("The best parameters are %s with a score of %0.5f"
          % (clf.best_params_, clf.best_score_))

    sorted(clf.cv_results_.keys())
    mean_score = clf.cv_results_['mean_test_score']
    std_score =  clf.cv_results_['std_test_score'][ np.argmax(mean_score)]
    best_params = clf.cv_results_['params'][ np.argmax(mean_score)]

    print "*** cross validation knn clustering ***"
    print best_params['n_neighbors']
    print ("accuracy is: %0.5f" %get_accuracy(y_test, Y_pred))
    print "Grid score function: ", clf.score(X_test, y_test)

    y_pred = clf.predict(X_test)
    # svc.fit(X_train, y_train)
    print max(mean_score)
    print std_score

    print "*** cross validation for knn with best param ***"
    # print ("cv = %d, accuracy = %.5f +/- %.5f" %(n_splits, np.maximum(mean_score), std_score))

    # print ("accuracy is: " + str(get_accuracy(y_test, Y_pred)))
    # knn_new = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'])

    print classification_report(y_test, y_pred)
    return mean_score, std_score




def measure_feature_importance(X_train, y_train):
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import SelectFromModel
    clf = ExtraTreesClassifier(n_estimators=200)
    clf = clf.fit(X_train, y_train)

    clf = ExtraTreesClassifier(n_estimators=200)
    clf = clf.fit(X_train, y_train)
    features = pd.DataFrame()
    features['feature'] = X_train.columns
    features['importance'] = clf.feature_importances_

    features = pd.DataFrame()
    features['feature'] = X_train.columns
    features['importance'] = clf.feature_importances_
    print "*** feature importances are ranked ***"
    print features.sort_values(by=['importance'], ascending=False)

    return features

def cross_validation_svm(X_train, y_train, X_test, y_test):
    '''
    select best parameters, run svc and return cross validated
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return: mean, std
    '''
    dataframe = pd.DataFrame()

    cc, gg = np.meshgrid( np.logspace(-4, 1, 25), np.logspace(-4, 1, 25))
    dataframe['C'] = cc.flatten()
    dataframe['gamma'] = gg.flatten()

    tuned_parameters = {'kernel': ['rbf','sigmoid'],'C': np.logspace(-4, 1, 25), 'gamma': np.logspace(-4, 1, 25)}

    clf = GridSearchCV(SVC(), tuned_parameters, cv = 5,
                       scoring='accuracy')

    clf.fit(X_train, y_train)
    print("The best parameters are %s with a score of %0.5f"
          % (clf.best_params_, clf.best_score_))

    sorted(clf.cv_results_.keys())
    mean_score = clf.cv_results_['mean_test_score']
    std_score =  clf.cv_results_['std_test_score'][ np.argmax(mean_score)]
    best_params = clf.cv_results_['params'][ np.argmax(mean_score)]

    # visualize cross validation
    dataframe['mean score'] = clf.cv_results_['mean_test_score']
    dataframe = dataframe.pivot('C', 'gamma', 'mean score')
    sns.heatmap(dataframe)

    y_pred = clf.predict(X_test)
    # svc.fit(X_train, y_train)

    print "*** cross validation for svm with best param ***"

    # print ("accuracy is: " + str(get_accuracy(y_test, Y_pred)))

    print classification_report(y_test, y_pred)
    return mean_score, std_score


def svm_cluster(X_train, y_train, X_test, y_test):
    print "*** svm clustering ***"
    # ---- ---- ---- cross validation
    svc = SVC(C= 6.1896581889126097, gamma= 0.21544346900318845)
    svc.fit(X_train, y_train)
    Y_pred = svc.predict(X_test)
    # acc_svc = round(svc.score(X_train, y_test) * 100, 2)
    print ("accuracy is: %0.5f" %get_accuracy(y_test, Y_pred))
    print classification_report(y_test, Y_pred)

    data = pd.concat([X_train, X_test])
    labels = pd.concat([y_train, y_test])

    mean, std = cross_validation_model(svc, data, labels)

    return mean, std


def logistic_regression(X_train, y_train, X_test, y_test):
    '''

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return: mean, std for cross validation
    '''

    # X_train.shape, y_train.shape, X_test.shape, y_test.shape

    # Logistic Regression
    logreg = LogisticRegression()

    data = pd.concat([X_train, X_test])
    labels = pd.concat([y_train, y_test])
    # cross validation
    # single run
    logreg.fit(X_train, y_train)
    Y_pred = logreg.predict(X_test)

    mean, std = cross_validation_model(logreg, data, labels)
    print classification_report(y_test, Y_pred)


    # acc_log = round(logreg.score(X_train, y_train) * 100, 2)
    # acc_log = 73.25
    print "*** logistic regression ***"
    print ("accuracy is: %0.5f" % get_accuracy(y_test, Y_pred))

    return mean, std


def final_set(X_train, y_train, X_test, y_test):
    logistic_regression(X_train, y_train, X_test, y_test)
    svm_cluster(X_train, y_train, X_test, y_test)
    naive_bayesian_cluster(X_train, y_train, X_test, y_test)
    knn_cluster(X_train, y_train, X_test, y_test)
