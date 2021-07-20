import configparser
import math
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict, Counter
from utils import *
import re
import os
import datetime

def cleaning_string(string):
    string = string.strip()
    string = string.lower()
    string = string.replace("‘", "")
    string = string.replace("“", "")
    string = string.replace("’", "")
    string = string.replace("/", "")
    string = re.sub(r"\d+", "", string)
    hangul = re.compile('[^ \- a-zㅣA-Z]+')
    string = hangul.sub('', string)
    return string


def sampling_corpus(corpus_path, ratio):
    file_len = len(open(corpus_path, "r").readlines())
    with open(corpus_path, "r") as f:
        lines = random.sample(f.readlines(), math.ceil(file_len * float(ratio)))
    lines = [cleaning_string(line) for line in lines]
    return lines


class KeywordExtractor:
    DEBUG_PRINT = True
    def __init__(self, config):
        corpus_config = config["corpus"]
        tfidf_config = config["tfidf"]
        default_config = config["default"]
        self.save_config = config["save"]
        
        # set Corpus
        self.hscd_corpus_path = corpus_config["hscd_corpus_path"]
        self.coid_corpus_path = corpus_config["coid_corpus_path"]
        
        self.total_corpus_path = [self.hscd_corpus_path, self.coid_corpus_path]
        
        self.sampling_ratio = float(default_config["corpus_sampling_ratio"])
        self.sampled_corpus = sampling_corpus(self.coid_corpus_path, ratio=self.sampling_ratio)

        # set Tokenizer
        self.tokenizer = lambda x : re.split("[ \.,\\t]", x)

        # set Stopwords
        use_default_stopwords = bool(default_config["use_default_stopwords"])
        self.default_stops = []
        if use_default_stopwords:
            self.default_stops = reader(default_config["defalut_stop_path"])

        # set tf-idf stopwords
        self.tfidf_threshold = tfidf_config["tfidf_threshold"]
        self.tfidf_stop_words = self._set_tfidf_stopwords(
            self.sampled_corpus, threshold=self.tfidf_threshold, save_path=os.path.join(self.save_config["save_path"], "keywords/tfidf_stop_words.save")
        )

        self.total_stop_words = list(set(self.default_stops) | set(self.tfidf_stop_words))

        self.min_count_stop_words = self._set_min_count_stopwords(self.sampled_corpus, default_config["min_count"])
        self.total_stop_words = list(set(self.total_stop_words) | set(self.min_count_stop_words))

    @time_printer(DEBUG_PRINT)
    def _set_tfidf_stopwords(self, sampled_corpus, threshold, save_path=None):
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

    @time_printer(DEBUG_PRINT)
    def _set_min_count_stopwords(self, sampled_corpus, min_count, save_path = None):
        min_count_stop_words = []
        cnt_dict = defaultdict(lambda : 0)
        for line in sampled_corpus:
            line = cleaning_string(line)
            tokenized_input = self.tokenizer(line)
            keywords = []
            for word in tokenized_input:
                # if word.pos == 'N' & word not in stop_words:
                if word not in self.total_stop_words:
                    cnt_dict[word] += 1

        for k, v in cnt_dict.items():
            if v < int(min_count):
                min_count_stop_words.append(k)

        if save_path is not None:
            writer(tfidf_stops, save_path)

        return min_count_stop_words

    def get_keywords(self, line):
        line = cleaning_string(line)
        tokenized_input = self.tokenizer(line)
        keywords = []
        for word in tokenized_input:
            # if word.pos == 'N' & word not in stop_words:
            if word not in self.total_stop_words:
                keywords.append(word)
        return keywords

    @time_printer(DEBUG_PRINT)
    def save_keywords(self, corpus_path, save_path, sep=","):
        ret_keywords = []
        empty_index = []
        ret_meta = dict()
        
        _eachline_keyword_count = 0
        _lines = 0
        _word_count_dict = defaultdict(lambda : 0)

        sf = open(save_path, "w")
        with open(corpus_path, "r") as f:
            for _lines, line in enumerate(f.readlines()):
                _lines += 1
                key = str(_lines) # key 추가 필요함
                keywords = self.get_keywords(line)
                for _w in keywords:
                    _word_count_dict[_w] += 1
                
                _eachline_keyword_count += len(keywords)
                if len(keywords) == 0:
                    empty_index.append(key)
                
                _tmp_string = key + "|" + sep.join(keywords) + "\n"
                sf.write(_tmp_string)
        sf.close()

        ret_meta['process_date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        ret_meta['total_lines'] = _lines
        ret_meta['extraction'] = f"{_lines-len(empty_index)}"
        ret_meta['empty_keywords_ratio'] = f"{len(empty_index)/_lines*100:.2f}% ({len(empty_index)}/{_lines})"
        
        ret_meta['tfidf_sampling_ratio'] = f"{self.sampling_ratio*100}%"
        ret_meta['#_of_stopwords'] = len(self.total_stop_words)

        ret_meta['total_keywords_cnt'] = f"{_eachline_keyword_count}"
        ret_meta['average_keywords_cnt'] = f"{_eachline_keyword_count/_lines:.2f}"

        ret_meta['empty_indices'] = empty_index
        ret_meta['counter_dict'] = Counter(_word_count_dict)

        return ret_meta

    def run(self):
        save_root_path = self.save_config['save_path']
        for corpus_path in self.total_corpus_path:
            save_path = os.path.join(save_root_path, 'keywords')
            file_name = corpus_path.split(os.sep)[-1]
            meta_infos = self.save_keywords(corpus_path, os.path.join(save_path, file_name + ".extraction"))
            writer(meta_infos, os.path.join(save_path, file_name + ".meta.dict"), pkl = True)

            meta_string = []
            for k,v in meta_infos.items():
                if k != 'counter_dict':
                    if type(v) == list:
                        v = ",".join([str(_v) for _v in v])
                    meta_string.append(" : ".join([k, str(v)]))
            
            writer(meta_string, os.path.join(save_path, file_name + ".meta"))
            break

if __name__ == "__main__":
    # Input  : config file
    # Output : 결과 파일         config file path + '.extraction' (save in same path)
    #          메타 정보 딕셔너리  config file path + '.meta.dict'  (save in same path)
    #          메타 정보 텍스트    config file path + '.meta'       (save in same path)

    
    config = configparser.ConfigParser()
    config.read("./conf/keyword_config.conf")
    ck = KeywordExtractor(config)
    ck.run()

    
