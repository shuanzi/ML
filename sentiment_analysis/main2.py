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


if __name__ == "__main__":

    X_all = traindata + testdata # Combine both to fit the TFIDF vectorization.
    lentrain = len(traindata)

    tfv.fit(X_all) # This is the slow part!
    X_all = tfv.transform(X_all)

X = X_all[:lentrain] # Separate back into training and test sets.
X_test = X_all[lentrain:]