import re
import nltk

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer

from utils.replacers import RegexReplacer
from utils.file_utils import get_all_file_path, _open

cachedStopWords = stopwords.words("english")
porter_stemmer = PorterStemmer()


class TextPreProcess(object):
    def __init__(self, _text):
        self.text = _text
        self.PunktTokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
        self.rc = re.compile(r"\<.*?\>")
        self.wordnet_lemmatizer = WordNetLemmatizer()
        pattern = r'[\d.,]+|[A-Z][.A-Z]+\b\.*|\w+|\S'
        self.tokenizer = RegexpTokenizer(pattern)
        self.replacer = RegexReplacer()

    def RemoveHTML(self):
        return [BeautifulSoup(sentence, "lxml").get_text() for sentence in self.text]

    def SplitPhase(self):
        return self.PunktTokenizer.tokenize(self.text)

    def ReplaceAbbre(self):
        return [self.replacer.replace(sentence) for sentence in self.text]

    def SplitSent(self):
        return [self.tokenizer.tokenize(sentence) for sentence in self.text]

    def lemma(self, tags):
        WORD = []
        for word, tag in tags:
            wntag = tag[0].lower()
            wntag = wntag if wntag in ['a', 'r', 'n', 'v', 'n', 's'] else None
            if not wntag:
                lemma = word
            else:
                lemma = self.wordnet_lemmatizer.lemmatize(word, wntag)

            WORD.append(lemma)
        return WORD

    def Lemmatizer(self):
        return [self.lemma(nltk.pos_tag(sentence)) for sentence in self.text]

    def CleanWords(self, sentence):
        stops = cachedStopWords
        return [word.lower() for word in sentence if len(word) >= 3 and word.isalpha() and not word in stops]

    def CleanSentences(self):
        return [self.CleanWords(sentence) for sentence in self.text]

    def ToStr(self):
        str = ""
        for sentence in self.text:
            for word in sentence:
                str += (word + " ")
        return str[:-1]

    def process(self):
        self.text = self.SplitPhase()
        self.text = self.ReplaceAbbre()
        self.text = self.SplitSent()
        self.text = self.Lemmatizer()
        self.text = self.CleanSentences()
        self.text = self.ToStr()
        return self.text

    def Print(self):
        print(self.text)


def wash_to_csv(argv, isTraining):
    pos_files = argv[0]
    neg_files = argv[1]
    unsup_files = argv[2]

    csv_file_name = "test.csv"
    if isTraining:
        csv_file_name = "trian.csv"

    lines = []
    lines.append("sentiment,comments\n")
    for file in pos_files:
        sentiment_value = 1
        content = _open(file)
        text_processor = TextPreProcess(content)
        clean_text = text_processor.process()
        line = str(sentiment_value) + "," + clean_text + "\n"
        lines.append(line)
        print(file + ", done!")

    for file in neg_files:
        sentiment_value = -1
        content = _open(file)
        text_processor = TextPreProcess(content)
        text_processor.Print()
        clean_text = text_processor.process()
        line = str(sentiment_value) + "," + clean_text + "\n"
        lines.append(line)
        print(file + ", done!")

    for file in unsup_files:
        sentiment_value = 0
        content = _open(file)
        text_processor = TextPreProcess(content)
        text_processor.Print()
        clean_text = text_processor.process()
        line = str(sentiment_value) + "," + clean_text + "\n"
        lines.append(line)
        print(file + ", done!")

    fo = open(csv_file_name, "w")
    fo.writelines(lines)


def process_training_data():
    pos_root_dir = "../resource/imdb_data/train/pos"
    neg_root_dir = "../resource/imdb_data/train/neg"
    unsup_root_dir = "../resource/imdb_data/train/unsup"

    pos_files = get_all_file_path(pos_root_dir)
    neg_files = get_all_file_path(neg_root_dir)
    unsup_files = get_all_file_path(unsup_root_dir)
    wash_to_csv([pos_files, neg_files, unsup_files, "pos"], True)


def process_test_data():
    pos_root_dir = "../resource/imdb_data/test/pos"
    neg_root_dir = "../resource/imdb_data/test/neg"

    pos_files = get_all_file_path(pos_root_dir)
    neg_files = get_all_file_path(neg_root_dir)
    wash_to_csv([pos_files, neg_files, "", "pos"], False)


if __name__ == "__main__":
    process_training_data()
    process_test_data()
