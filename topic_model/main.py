import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic #%d:" % topic_idx
        print " ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]])
    print model.components_


class Preprocessing():
    def __init__(self, train_data_path, test_data_path):
        self.train = pd.read_csv(train_data_path)
        self.test = pd.read_csv(test_data_path)
        self.y_train = self.train['sentiment']
        self.y_test = self.test['sentiment']
        self.feature_names = []

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

        self.feature_names = vectorizer.get_feature_names()

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

    def lda(self, feature_names):
        n_topics = 30
        lda = LatentDirichletAllocation(n_topics=n_topics,
                                        max_iter=50,
                                        learning_method='batch')

        lda.fit(self.X_train)

        print_top_words(lda, feature_names, n_topics)


if __name__ == "__main__":
    process = Preprocessing("../resource/train.csv", "../resource/test.csv")
    process.build_data()
    X_train, y_train, X_test, Y_test = process.vec()
    clf = classify(X_train, y_train, X_test, Y_test)
    clf.lda(process.feature_names)
