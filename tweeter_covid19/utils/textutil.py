"""
 -- author : Anish Basnet
 -- email : anishbasnetworld@gmail.com
 -- date : 11/14/2019
"""
import re
import string
from string import punctuation

from nltk import word_tokenize, PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
from collections import OrderedDict
from tweeter_covid19.utils.utils import split_word_into_letters


def filter_text(text):
    """
    This methods clean the documents in the text level.
    :param text: String -> 'document text'
    :return: String -> 'clean text'
    """
    table = str.maketrans('', '', punctuation)
    x = [w.translate(table) for w in word_tokenize(text)]
    x = [w for w in x if w.isalpha()]
    x = [w for w in x if (len(w) > 1)]
    x = [w for w in x if wordnet.synsets(w)]
    clean_text = ''
    for word in x:
        clean_text = clean_text + ' ' + word
    return clean_text


def preprocess_documents(documents):
    """
    This method cleans the documents into the clean text documents.
    :param documents: List
    :return: List
    """
    filter_list = list()
    for x in documents:
        table = str.maketrans('', '', punctuation)
        x = [w.translate(table) for w in word_tokenize(x)]
        x = [w for w in x if w.isalpha()]
        x = [w for w in x if (len(w) > 1)]
        x = [w for w in x if wordnet.synsets(w)]
        filter_list.append(' '.join(x).lower())
    return filter_list


def create_vocab(texts, tokn):
    """
    This method is used to create the vocabulary fdef preprocess_nepali_documents(documents, stop_words):
    words = []
    for document in documents:
        tokens = document.split(' ')
        for token in tokens:
            if token not in stop_words and len(token) > 1 and not re.search(pattern='[०१२३४५६७८९]', string=token):
                print("found, ", token)

        exit(0)rom the strings.
    :param texts: List-> ['string']
    :return: List->['word',.....,'word']
    """

    count_vector = CountVectorizer(token_pattern=tokn)
    count = count_vector.fit_transform(texts)
    frequency = zip(count_vector.get_feature_names(), count.sum(axis=0).tolist()[0])
    word_with_freq = sorted(frequency, key=lambda x: -x[1])
    return word_with_freq


def get_filtered_words(words=None):
    """
    This function gives filtered words.
    :param words: list -> 1d list
    :return: list -> 1d list
    """
    if words is None:
        return None
    stemmer = PorterStemmer()
    lemma = WordNetLemmatizer()
    stem_words = [stemmer.stem(lemma.lemmatize(word)) for word in words]
    filter_words = list()
    for word in stem_words:
        temp = []
        for index, _word in enumerate(stem_words):
            if word == _word:
                temp.append(words[index])
        filter_words.append(sorted(temp)[0])
    return list(set(filter_words))


def add_word_with_freq(words=None, word_with_freq=None):
    if word_with_freq is None or words is None:
        return None
    stemmer = PorterStemmer()
    lemma = WordNetLemmatizer()
    _word_with_freq = []
    for word in words:
        stem_word = stemmer.stem(lemma.lemmatize(word))
        total_freq = 0
        for _word, frequency in word_with_freq:
            _stem_word = stemmer.stem(lemma.lemmatize(_word))
            if stem_word == _stem_word:
                total_freq += frequency
        _word_with_freq.append((word, total_freq))
    return sorted(_word_with_freq, key=lambda x: -x[1])


def preprocess_nepali_documents(documents, stop_words, verbose=False):
    words = []
    for index, document in enumerate(documents):
        if verbose:
            print("Pre-processing document {} - Completed Successfully. Remaining : {}/{} ."
                  .format(index, index, len(documents)))
        tokens = document.split(' ')
        for token in tokens:
            if re.search(pattern='।|’|,|‘', string=token):
                match_regular_expression = re.findall(pattern='।|’|,|‘', string=token)
                for matched_string in match_regular_expression:
                    token = re.sub(matched_string, '', token).strip()
            if token not in stop_words and len(token) > 1 and not re.search(pattern='[०१२३४५६७८९]', string=token):
                words.append(token)
    words = list(OrderedDict.fromkeys(words))
    words_with_useage = []
    for word in words:
        words_with_useage.append((word, split_word_into_letters(word)))
    return words_with_useage


def match_letters(source, destination, pairs=None):
    for src_letter, dest_letter in zip(source, destination):
        if src_letter != dest_letter:
            status = False
            for pair in pairs:
                letters = pair.strip().split(',')
                count = 0
                for letter in letters:
                    if letter.strip() == src_letter or letter.strip() == dest_letter:
                        count += 1
                        if count == 2:
                            status = True
                            break
                if status:
                    break
            if not status:
                return False
    return True


def filter_rasuwa_dirga(source, destination, pairs=None):
    src_len = len(source)
    dest_len = len(destination)
    if src_len == dest_len:
        return match_letters(source, destination, pairs=pairs)
    else:
        return False


def process_rasuwa_dirga(tokens=None, pairs=None, verbose=False):
    if tokens is None or pairs is None:
        return None
    for token in tokens:
        for _token in tokens:
            if not token == _token:
                result = filter_rasuwa_dirga(token[1], _token[1], pairs=pairs)
                if result:
                    tokens.remove(_token)
                    if verbose:
                        print("Match word : {} - {} . Total Tokens : {} . ".format(token[0], _token[0],
                                                                                   len(tokens)))
    return tokens


# TODO not completely refactored yet.
def filter_text(sentence):
    string.punctuation
    '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    sentence.strip(string.punctuation)
    cleaned_fullstop = re.sub(r'[।ः|०-९]', '', str(sentence))
    # clean_text = re.sub(r'[^\w\s]', '', str(cleaned_fullstop))
    return ' '.join(re.findall(r'[\u0900-\u097F]+', str(cleaned_fullstop), re.IGNORECASE))
