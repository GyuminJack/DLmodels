import math
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import re


def cleaning_string(string):
    string = string.strip()
    string = string.lower()
    string = string.replace("‘", "")
    string = string.replace("“", "")
    string = string.replace("’", "")
    string = re.sub(r"\d+", "", string)
    return string


def sampling_corpus(corpus_path, ratio):
    file_len = len(open(corpus_path, "r").readlines())
    with open(corpus_path, "r") as f:
        lines = random.sample(f.readlines(), math.ceil(file_len * float(ratio)))
    lines = [cleaning_string(line) for line in lines]
    return lines


def writer(result, file_path):
    try:
        with open(file_path, "w") as f:
            for i in result:
                f.write(i + "\n")
    except Exception as e:
        print(f"Can't save : {e}")


def reader(path):
    ret = []
    with open(path, "r") as f:
        for line in f.readlines():
            ret.append(line.strip())
    return ret


class KeywordExtractor:
    def __init__(self, config):
        corpus_config = config["corpus"]
        tfidf_config = config["tfidf"]
        default_config = config["default"]
        save_config = config["save"]

        # set Corpus
        self.corpus_path = corpus_config["corpus_path"]
        self.tf_idf_corpus = sampling_corpus(self.corpus_path, ratio=tfidf_config["tfidf_sampling_ratio"])

        # set Tokenizer
        self.tokenizer = lambda x: re.split("[ \.,\\t]", x)

        # set Stopwords
        use_default_stopwords = bool(default_config["use_default_stopwords"])
        self.default_stops = []
        if use_default_stopwords:
            self.default_stops = reader(default_config["defalut_stop_path"])

        # set tf-idf stopwords
        self.tfidf_stop_words = self._set_tfidf_stopwords(
            self.tf_idf_corpus, default_config["min_count"], threshold=tfidf_config["tfidf_threshold"], save_path=tfidf_config["save_path"]
        )

    def _set_tfidf_stopwords(self, sampled_corpus, min_count, threshold, save_path=None):
        tfidf_stops = []
        vectorizer = TfidfVectorizer(stop_words=self.default_stops)

        sp_matrix = vectorizer.fit_transform(sampled_corpus)
        word2id = defaultdict(lambda: 0)

        for idx, feature in enumerate(vectorizer.get_feature_names()):
            word2id[feature] = idx

        for i, sent in enumerate(sampled_corpus):
            for token in self.tokenizer(sent):
                if sp_matrix[i, word2id[token]] < float(threshold):
                    tfidf_stops.append(token)

        tfidf_stops = list(set(tfidf_stops))
        if save_path is not None:
            writer(tfidf_stops, save_path)

        return tfidf_stops

    def get_keywords(self, sep="|"):
        ret_keywords = []
        stop_words = self.default_stops + self.tfidf_stop_words
        with open(self.corpus_path, "r") as f:
            for line in f.readlines():
                line = cleaning_string(line)
                tokenized_input = self.tokenizer(line)
                keywords = []
                for word in tokenized_input:
                    # if word.pos == 'N' & word not in stop_words:
                    if word not in stop_words:
                        keywords.append(word)
                ret_keywords.append(sep.join(keywords))
        return ret_keywords

    def push_to_db(self, dbconn):
        pass


if __name__ == "__main__":
    import configparser

    config = configparser.ConfigParser()
    config.read("./conf/keyword_config.conf")
    ck = KeywordExtractor(config)
    writer(ck.get_keywords(), config["save"]["file_path"])
