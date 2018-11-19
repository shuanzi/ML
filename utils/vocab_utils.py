import jieba
from nltk.stem.porter import PorterStemmer

STOP_WORDS = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any",
              "are", "as", "at", "be", "because", "been", "before", "being", "below", "between",
              "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each",
              "few", "for", "from", "further", "had", "has", "have", "having", "he", "he’d",
              "he’ll", "he’s", "her", "here", "here’s", "hers", "herself", "him", "himself", "his",
              "how", "how’s", "I", "I’d", "I’ll", "I’m", "I’ve", "if", "in", "into", "is", "it",
              "it’s", "its", "itself", "let’s", "me", "more", "most", "my", "myself", "nor", "of",
              "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
              "over", "own", "same", "she", "she’d", "she’ll", "she’s", "should", "so", "some",
              "such", "than", "that", "that’s", "the", "their", "theirs", "them", "themselves",
              "then", "there", "there’s", "these", "they", "they’d", "they’ll", "they’re",
              "they’ve", "this", "those", "through", "to", "too", "under", "until", "up", "was",
              "very", "we", "we’d", "we’ll", "we’re", "we’ve", "were", "what", "what’s", "when",
              "when’s", "where", "where’s", "which", "while", "who", "who’s", "whom", "why",
              "why’s", "with", "would", "you", "you’d", "you’ll", "you’re", "you’ve", "your",
              "yours", "yourself", "yourselves"]

porter_stemmer = PorterStemmer()

def remove_stop_words(wordList, stopwords):
    return [w for w in wordList if w not in stopwords]

def text_parse(text):
    cutted = jieba.cut(text)
    list_of_word = remove_stop_words(cutted, STOP_WORDS)
    stemmered_list = []
    for word in list_of_word:
        stemmered_word = porter_stemmer.stem(word)
        if is_alpha(word):
            stemmered_list.append(stemmered_word.encode('utf-8'))
    return stemmered_list


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def is_alpha(string):
    return string.isalpha()
