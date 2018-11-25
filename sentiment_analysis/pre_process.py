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
        """ remove HTML tags """
        return [BeautifulSoup(sentence, "lxml").get_text() for sentence in self.text]

    def SplitPhase(self):
        """ split paragraph to sentence """
        return self.PunktTokenizer.tokenize(self.text)

    def ReplaceAbbre(self):
        """ Replace abbreviation """
        return [self.replacer.replace(sentence) for sentence in self.text]

    def SplitSent(self):
        """ split sentence to words """
        return [self.tokenizer.tokenize(sentence) for sentence in self.text]

    def lemma(self, tags):
        """ lemmatizer for tagged words """
        WORD = []
        for word, tag in tags:
            wntag = tag[0].lower()
            wntag = wntag if wntag in ['a', 'r', 'n', 'v', 'n', 's'] else None
            if not wntag:
                lemma = word
            else:
                lemma = self.wordnet_lemmatizer.lemmatize(word, wntag)

            # stemmered_word = porter_stemmer.stem(lemma)
            WORD.append(lemma)
        return WORD

    def Lemmatizer(self):
        """ Lemmatizer words use WordNet """
        return [self.lemma(nltk.pos_tag(sentence)) for sentence in self.text]

    def CleanWords(self, sentence):
        """ remove len < 3 and non alpha and lowercase """
        stops = cachedStopWords
        return [word.lower() for word in sentence if len(word) >= 3 and word.isalpha() and not word in stops]

    def CleanSentences(self):
        """ clean sentences """
        return [self.CleanWords(sentence) for sentence in self.text]

    def ToStr(self):
        str = ""
        for sentence in self.text:
            for word in sentence:
                str += (word + " ")
        return str[:-1]

    def process(self):
        """ Remove HTML tags, Replace Abbre, Split into words
            if use word2vector, should not use ``stopwords ``
        """
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
    files = argv[0]
    sentiment = argv[1]
    sentiment_value = argv[2]

    if isTraining:
        csv_file_name = sentiment + ".tsv"
        lines = []
        lines.append("sentiment\tcomments\n")
        for file in files:
            content = _open(file)
            text_processor = TextPreProcess(content)
            text_processor.Print()
            clean_text = text_processor.process()
            # text_processor.Print()
            line = str(sentiment_value) + "\t" + clean_text + "\n"
            lines.append(line)
            print(file + ", done!")

        fo = open(csv_file_name, "w")
        fo.writelines(lines)
    else:
        csv_file_name = sentiment + "_test.tsv"
        lines = []
        lines.append("sentiment\tomments\n")
        for file in files:
            content = _open(file)
            line = str(sentiment_value) + "\t" + content + "\n"
            lines.append(line)
            print(file + ", done!")

        fo = open(csv_file_name, "w")
        fo.writelines(lines)


def process_training_data():
    pos_root_dir = "../resource/imdb_data/train/pos"
    neg_root_dir = "../resource/imdb_data/train/neg"
    unsup_root_dir = "../resource/imdb_data/train/unsup"
    pos_files = get_all_file_path(pos_root_dir)
    print(pos_files)
    neg_files = get_all_file_path(neg_root_dir)
    print(neg_files)
    unsup_files = get_all_file_path(unsup_root_dir)
    print(unsup_files)
    argvs = []
    argvs.append([pos_files, "pos", 1])
    argvs.append([neg_files, "neg", -1])
    argvs.append([unsup_files, "unsup", 0])
    for arg in argvs:
        wash_to_csv(arg, True)


def process_test_data():
    pos_root_dir = "../resource/imdb_data/test/pos"
    neg_root_dir = "../resource/imdb_data/test/neg"

    pos_files = get_all_file_path(pos_root_dir)
    print(pos_files)
    neg_files = get_all_file_path(neg_root_dir)
    print(neg_files)
    argvs = []
    argvs.append([pos_files, "pos", 1])
    argvs.append([neg_files, "neg", -1])
    for arg in argvs:
        wash_to_csv(arg, True)


if __name__ == "__main__":
    # process_training_data()
    process_test_data()
