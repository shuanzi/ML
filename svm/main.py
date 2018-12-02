from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV


def load_data():
    keeplandData = pd.read_csv('/Users/daixiquan/Documents/workout/decision_tree/resource/keepland_user_profile.csv')
    # keeplandData = pd.read_csv('/Users/daixiquan/Documents/workout/decision_tree/resource/keepland_user_profile2.csv')
    keeplandData.head()

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
    param_test1 = {'C': [0.01,10]}
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
    clf = SVC(kernel='linear', C=0.1)
    searchBestParams(clf)
    # try_different_method(clf, "kernel='linear'")


