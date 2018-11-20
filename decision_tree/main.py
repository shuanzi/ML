from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report

import pandas as pd


def load_data():
    keeplandData = pd.read_csv('/Users/daixiquan/Documents/workout/decision_tree/resource/keepland_user_profile.csv')
    keeplandData.head()
    # keeplandData.info()

    X = keeplandData[
        ['bookcount', 'trainingcount', 'age', 'gender', 'goal', 'trainingtimes', 'trainingdays', 'allduration',
         'allcalorie', 'maxcomboday', 'kg', 'bmi']]

    y = keeplandData['bought_package']
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=33)
    vec = DictVectorizer(sparse=False)
    X_train = vec.fit_transform(X_train.to_dict(orient='record'))
    X_test = vec.fit_transform(X_test.to_dict(orient='record'))

    return X_train, X_test, Y_train, Y_test, vec.feature_names_


x_train, x_test, y_train, y_test, feature_names = load_data()


def searchBestParams(clf):
    # 首先对n_estimators进行网格搜索
    param_test1 = {'n_estimators': range(10, 71, 10)}
    gsearch1 = GridSearchCV(clf, param_grid=param_test1, scoring='roc_auc', cv=5)
    gsearch1.fit(x_train, y_train)
    print(gsearch1.best_estimator_, gsearch1.best_score_)


def try_different_method(clf, title):
    clf.fit(x_train, y_train)
    result = clf.predict(x_test)
    score = clf.score(x_test, y_test)
    print(title)
    print(score)
    print(classification_report(y_test, result, target_names=["not buy", "buy"]))
    print("\n")


if __name__ == "__main__":
    dt = DecisionTreeClassifier(criterion="entropy", max_depth=5)
    try_different_method(dt, "DecisionTree")

    rf = ensemble.RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                         max_depth=8, max_features='sqrt', max_leaf_nodes=None,
                                         min_impurity_decrease=0.0, min_impurity_split=None,
                                         min_samples_leaf=20, min_samples_split=100,
                                         min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None,
                                         oob_score=False, random_state=10, verbose=0, warm_start=False)
    try_different_method(rf, "RandomForest")
    # searchBestParams(rf)

    ada = ensemble.AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
                                      learning_rate=1.0, n_estimators=40, random_state=None)
    try_different_method(ada, "AdaBoost")
    # searchBestParams(ada)
