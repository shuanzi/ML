import pandas as pd
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer as TFIV
from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_extraction.text import CountVectorizer


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
        grid_values = {'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}

        clf = GridSearchCV(LR(penalty='l2', dual=True, random_state=0),
                           grid_values, scoring='roc_auc', cv=20, n_jobs=4)
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


if __name__ == "__main__":
    process = Preprocessing("../resource/train2.csv", "../resource/test.csv")
    process.build_data()
    X_train, y_train, X_test, Y_test = process.tfidf(ngram=4)
    # X_train, y_train, X_test, Y_test = process.vec()
    clf = classify(X_train, y_train, X_test, Y_test)
    # clf.mnb()
    clf.LR()
