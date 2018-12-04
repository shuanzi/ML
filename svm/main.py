import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def load_data():
    keeplandData = pd.read_csv('../data/keepland_user_profile.csv')
    keeplandData.head()

    X = keeplandData[
        ['bookcount', 'trainingcount', 'age', 'goal', 'trainingtimes', 'trainingdays', 'allduration',
         'allcalorie', 'maxcomboday', 'kg', 'bmi']]

    y = keeplandData['bought_package']
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=33)

    scaler1 = StandardScaler()
    scaler1.fit(X_train)
    X_train = scaler1.fit_transform(X_train)

    scaler2 = StandardScaler()
    scaler2.fit(X_test)
    X_test = scaler2.fit_transform(X_test)

    return X_train, X_test, Y_train, Y_test


x_train, x_test, y_train, y_test = load_data()


def searchBestParams(clf, score):
    # 首先对n_estimators进行网格搜索
    tuned_parameters = [{'kernel': ['poly'], 'gamma': [1e-3, 1e-4],'C': [1, 2, 4, 8, 10]}]
    # tuned_parameters = [{'kernel': ['poly'], 'C': [1, 2, 4, 8, 10]}]
    # scores = ['precision', 'recall']
    gsearch1 = GridSearchCV(clf, tuned_parameters, scoring='%s_macro' % score, cv=5)
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
    # clf = SVC(C=15, cache_size=200, class_weight=None, coef0=0.0,
    #           decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
    #           kernel='linear', max_iter=-1, probability=False, random_state=None,
    #           shrinking=True, tol=0.001, verbose=False)
    # try_different_method(clf, "kernel='linear'")

    # clf = SVC(C=2, cache_size=200, class_weight=None, coef0=0.0,
    #           decision_function_shape='ovr', degree=3, gamma=0.0001, kernel='rbf',
    #           max_iter=-1, probability=False, random_state=None, shrinking=True,
    #           tol=0.001, verbose=False)
    # # searchBestParams(clf, "precision")
    # try_different_method(clf, "kernel='rbf'")

    # clf = SVC(C=8, cache_size=200, class_weight=None, coef0=0.0,
    #           decision_function_shape='ovr', degree=3, gamma=0.001, kernel='poly',
    #           max_iter=-1, probability=False, random_state=None, shrinking=True,
    #           tol=0.001, verbose=False)
    # # searchBestParams(clf, "precision")
    # try_different_method(clf, "kernel='poly'")

    clf = SVC(C=8, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape='ovr', degree=3, gamma=0.001, kernel='poly',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False)
    # searchBestParams(clf, "precision")
    try_different_method(clf, "kernel='poly'")
