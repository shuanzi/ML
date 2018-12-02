import nltk

nltk.download()
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.model_selection import cross_val_score
import numpy as np

stops = set(stopwords.words("english"))


def review_to_words(review):
    r = BeautifulSoup(review).get_text()
    letters = re.sub("[^a-zA-Z]", " ", r)
    words = letters.lower().split()
    words = [w for w in stops if not w in stops]
    words = " ".join(words)

    return words


if __name__ == "__main__":
    train = pd.read_csv(", ", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv(", ", header=0, delimiter="\t", quoting=3)

    num_review = train["review"].size
    clean_train = []
    for i in range(0, num_review):
        clean_train.append(review_to_words(train["review"][i]))

    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None,
                                 stop_words=None, max_features=5000)
    tran_data_f = vectorizer.fit_transform(clean_train)
    tran_data_f = tran_data_f.toarray()

    model_NB = MNB()
    model_NB.fit(tran_data_f, train["sentiment"])

    score = np.mean(
        cross_val_score(model_NB, tran_data_f, train["sentiment"], cv=20, scoring="roc_auc"))
    print("MultinomialNB score: %s" % score)
