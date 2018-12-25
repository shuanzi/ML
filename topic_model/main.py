import pandas as pd
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer as TFIV
from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic #%d:" % topic_idx
        print " ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]])
    print
    print model.components_

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

        tf = vectorizer.fit_transform(X_all)

        n_topics = 30
        lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=50, learning_method='batch')
        lda.fit(tf)

        n_top_words=20
        tf_feature_names = vectorizer.get_feature_names()
        print_top_words(lda, tf_feature_names, n_top_words)

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
    process = Preprocessing("../data/train.csv", "../data/test.csv")
    process.build_data()
    X_train, y_train, X_test, Y_test = process.vec()
    # clf = classify(X_train, y_train, X_test, Y_test)
    # clf.mnb()
    # clf.LR()
