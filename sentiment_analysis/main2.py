from sklearn.feature_extraction.text import TfidfVectorizer as TFIV
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer as TFIV
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier as SGD
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import xgboost as xgb
import pickle
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import CountVectorizer


class Preprocessing():
    def __init__(self, train_data_path, test_data_path, Unlabeled=None):
        self.train = pd.read_csv(train_data_path)
        self.test = pd.read_csv(test_data_path)
        self.y_train = self.train['sentiment']
        self.y_test = self.test['sentiment']

    def commtent_to_word_list(self, comments):
        words = comments.lower().split()

        return (words)

    def build_data(self):
        self.traindata = []
        for i in range(0, len(self.train['comments'])):
            self.traindata.append(
                " ".join(self.commtent_to_word_list(self.train['comments'][i])))
        self.testdata = []
        for i in range(0, len(self.test['comments'])):
            self.testdata.append(
                " ".join(self.commtent_to_word_list(self.test['comments'][i])))

    def vec(self):
        vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None)

        X_all = self.traindata + self.testdata
        lentrain = len(self.traindata)

        vectorizer.fit(X_all)
        X_all = vectorizer.transform(X_all)

        self.X = X_all[:lentrain]
        self.X_test = X_all[lentrain:]

        print("vectorization data size: ", self.X.shape)
        return self.X, self.y_train, self.X_test, self.y_test

    def tfidf(self, ngram=2):
        tfv = TFIV(min_df=3, max_features=None,
                   strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                   ngram_range=(1, ngram), use_idf=1, smooth_idf=1, sublinear_tf=1,
                   stop_words='english')

        X_all = self.traindata + self.testdata
        lentrain = len(self.traindata)

        tfv.fit(X_all)
        X_all = tfv.transform(X_all)

        self.X = X_all[:lentrain]
        self.X_test = X_all[lentrain:]

        print("vectorization data size: ", self.X.shape)
        return self.X, self.y_train, self.X_test, self.y_test


class classify():
    def __init__(self, X_train, Y_Train, X_Test, Y_Test):
        self.X_train = X_train
        self.Y_Train = Y_Train
        self.X_Test = X_Test
        self.Y_Test = Y_Test

    def LR(self):
        """
        use LogisticRegression and GridSearchCV to find best parameters
        """
        # Decide which settings you want for the grid search.
        grid_values = {'C': [1e-3, 1e-2, 1e-1, 1, 2]}
        # grid_values = {'C': [1e-5,1e-4,1e-3,1e-2,1e-1]}

        clf = GridSearchCV(LR(penalty='l2', dual=True, random_state=0),
                           grid_values, scoring='roc_auc', cv=20, n_jobs=4)
        # Try to set the scoring on what the contest is asking for.
        # The contest says scoring is for area under the ROC curve, so use this.
        clf.fit(self.X_train, self.Y_Train)  # Fit the model.
        print("using LogisticRegression, Best: %f using %s" %
              (clf.best_score_, clf.best_params_))
        y_predict = clf.predict(self.X_Test)
        print(classification_report(self.Y_Test, y_predict))

    def mnb(self):
        clf = MNB()
        clf.fit(self.X_train, self.Y_Train)
        y_predict = clf.predict(self.X_Test)
        score = clf.score(self.X_Test, self.Y_Test)
        print("using mnb, score %s" % score)
        print(classification_report(self.Y_Test, y_predict))

    def SVMTest(self):
        """
        Pipeline+GridSearchCV
        """
        parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 0.005, 1e-3],
                       'C': [0.5, 1, 1.5, 2, 4]},
                      {'kernel': ['linear'], 'C': [1e-3, 1e-2, 0.1, 1]}]
        clf = GridSearchCV(
            SVC(probability=True),
            parameters,
            cv=5,
            scoring="roc_auc",
            n_jobs=4
        )
        clf.fit(self.X_train, self.Y_Train)
        print("using SVM, Best: %f using %s" %
              (clf.best_score_, clf.best_params_))
        self.best_clf = clf.best_estimator_
        return self.best_clf

    def sgd(self):
        # Regularization parameter
        # sgd_params = {'alpha': [ 0.18,0.17,0.19,0.185]}
        sgd_params = {'alpha': [1e-1, 0.5, 1, 1.5]}

        clf = GridSearchCV(SGD(max_iter=5, random_state=0, loss='modified_huber', n_jobs=4), sgd_params,
                           scoring='roc_auc', cv=20)  # Find out which regularization parameter works the best.

        clf.fit(self.X_train, self.Y_Train)
        print("using SGD, Best: %f using %s" %
              (clf.best_score_, clf.best_params_))
        self.best_clf = clf.best_estimator_
        return self.best_clf

    def sgdboot(self):
        cv_params = {'max_depth': [7, 9, 10], 'min_child_weight': [1, 3, 5]}
        ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed': 0, 'subsample': 0.8, 'colsample_bytree': 0.8,
                      'objective': 'binary:logistic'}
        clf = GridSearchCV(xgb.XGBClassifier(**ind_params),
                           cv_params,
                           scoring='accuracy', cv=5, n_jobs=4, verbose=True)
        clf.fit(self.X_train, self.Y_Train)
        print("using sgdboot, Best: %f using %s" %
              (clf.best_score_, clf.best_params_))
        # print(clf.grid_scores_)
        self.best_clf = clf.best_estimator_
        return self.best_clf


if __name__ == "__main__":
    process = Preprocessing("../resource/train.csv", "../resource/test.csv")
    process.build_data()
    # X_train, y_train, X_test, Y_test = process.tfidf(ngram=6)
    X_train, y_train, X_test, Y_test = process.vec()
    clf = classify(X_train, y_train, X_test, Y_test)
    # clf.mnb()
    clf.LR()
