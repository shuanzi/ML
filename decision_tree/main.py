from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import pandas as pd

if __name__ == "__main__":
    # http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt
    titanic = pd.read_csv('/Users/daixiquan/Documents/workout/decision_tree/resource/titanic.txt')
    titanic.head()

    titanic.info()

    X = titanic[['pclass', 'age', 'sex']]
    y = titanic['survived']

    # age 数据有确实，做一个默认取平均值的处理
    X['age'].fillna(X['age'].mean(), inplace=True)

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=33)
    vec = DictVectorizer(sparse=False)

    X_train = vec.fit_transform(X_train.to_dict(orient='record'))
    print(vec.feature_names_)
    X_test = vec.fit_transform(X_test.to_dict(orient='record'))

    # 决策树
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, Y_train)
    y_predict = dtc.predict(X_test)

    print(dtc.score(X_test, Y_test))
    print(classification_report(y_predict, Y_test, target_names=['died', 'survived']))

    X.info()

    # 随机森林
    rfc = RandomForestClassifier()
    rfc.fit(X_train, Y_train)
    rfc_y_predict = rfc.predict(X_test)

    print(rfc.score(X_train, Y_train))
    print(classification_report(rfc_y_predict, Y_train, target_names=["died", "survived"]))
