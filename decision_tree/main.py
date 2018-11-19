import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

import pandas as pd


def load_data():
    # titanic = pd.read_csv('/Users/daixiquan/Documents/workout/decision_tree/resource/titanic.txt')
    keeplandData = pd.read_csv('/Users/daixiquan/Documents/workout/decision_tree/resource/keepland_user_profile.csv')
    keeplandData.head()
    # keeplandData.info()

    # X = titanic[['bookcount', 'trainingcount', 'age', 'gender', 'goal', 'trainingduration', 'trainingcalorie',
    #              'trainingtimes', 'trainingdays', 'runningtimes', 'cyclingtimes', 'hikingtimes', 'allduration',
    #              'allcalorie', 'alltimes', 'alldays', 'traininggoal', 'difficulty', 'maxcomboday', 'kg', 'bmi',
    #              'joinedplans', 'plans']]
    X = keeplandData[
        ['bookcount', 'trainingcount', 'age', 'gender', 'goal', 'trainingtimes', 'trainingdays', 'allduration',
         'allcalorie', 'alltimes', 'alldays', 'traininggoal', 'difficulty', 'maxcomboday', 'kg', 'bmi']]
    y = keeplandData['bought_package']
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=33)
    vec = DictVectorizer(sparse=False)
    X_train = vec.fit_transform(X_train.to_dict(orient='record'))
    X_test = vec.fit_transform(X_test.to_dict(orient='record'))

    return X_train, X_test, Y_train, Y_test


x_train, x_test, y_train, y_test = load_data()


def try_different_method(clf, title):
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    result = clf.predict(x_test)
    plt.figure()
    plt.plot(np.arange(len(result)), y_test, 'go-', label='true value')
    plt.plot(np.arange(len(result)), result, 'ro-', label='predict value')
    plt.title('%s score: %f' % (title, score))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # dt = DecisionTreeClassifier(criterion="entropy",max_depth=5,)
    # try_different_method(dt, "decision tree")

    rf = ensemble.RandomForestRegressor(n_estimators=50,max_depth=5)  # 这里使用20个决策树
    try_different_method(rf, "random forest")
    #
    # ada = ensemble.AdaBoostRegressor(n_estimators=10)
    # try_different_method(ada, "ada boost")
