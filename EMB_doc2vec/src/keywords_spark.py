
import configparser
import math
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from pyspark.sql.functions import monotonically_increasing_id
from collections import defaultdict, Counter
import math
from pyspark.sql.functions import *
from utils import *
import re
import os
import datetime
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.functions import udf
from pyspark.sql.functions import lit
from pyspark.sql.types import *

def get_keywords(tokenizer, total_stop_words):
    def _get_keywords(line):
        keywords = []
        tokenized_input = tokenizer(line)
        for word in tokenized_input:
            # if word.pos == 'N' & word not in stop_words:
            if word not in total_stop_words:
                keywords.append(word)
        return ",".join(keywords)
    return F.udf(_get_keywords, StringType())

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


class KeywordExtractor:
    DEBUG_PRINT = True
    def __init__(self, config):
        corpus_config = config["corpus"]
        tfidf_config = config["tfidf"]
        default_config = config["default"]
        self.save_config = config["save"]
        
        # # set Corpus
        self.hscd_corpus_path = corpus_config["hscd_corpus_path"]
        self.coid_corpus_path = corpus_config["coid_corpus_path"]
        
        self.total_corpus_path = [self.hscd_corpus_path, self.coid_corpus_path]
        
        self.set_spark_session()
        self.set_df()

        self.sampling_ratio = float(default_config["corpus_sampling_ratio"])
        sampled_corpus = self.sampling_corpus(self.total_df, ratio=self.sampling_ratio)

        # set Tokenizer
        self.tokenizer = lambda x : re.split("[ \.,\\t]", x)

        # set Stopwords
        use_default_stopwords = bool(default_config["use_default_stopwords"])
        self.default_stops = []
        if use_default_stopwords:
            self.default_stops = reader(default_config["defalut_stop_path"])

        # set tf-idf stopwords
        self.tfidf_threshold = tfidf_config["tfidf_threshold"]
        self.total_stop_words = self.make_stopwords(sampled_corpus, self.tfidf_threshold, default_config["min_count"])

    def set_spark_session(self):
        from pyspark import SparkConf
        from pyspark.sql import SparkSession
        spark_conf = SparkConf().setAppName("ANGORA-LOCAL")
        self.spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()
        self.sc = self.spark.sparkContext
        self.spark.sparkContext.setLogLevel("ERROR")

    def set_df(self):
        columns = ['string']
        hscd_df = self.sc.textFile(self.hscd_corpus_path).map(lambda x: cleaning_string(x).split('|')).toDF(columns)
        hscd_df = hscd_df.withColumn('_key', lit(0))
        coid_df = self.sc.textFile(self.coid_corpus_path).map(lambda x: cleaning_string(x).split('|')).toDF(columns)
        coid_df = coid_df.withColumn('_key', lit(1))
        
        self.total_df = hscd_df.union(coid_df)
        w = Window.orderBy(lit(1))
        self.total_df = self.total_df.withColumn("_id", F.row_number().over(w))

    def sampling_corpus(self, corpus_df, ratio = 0.1):
        sampled_corpus = corpus_df.sample(ratio)
        return sampled_corpus

    def make_keywords_df(self):
        raw_string_column = 'string'
        total_corpus = self.total_df \
                .select('_id', '_key', F.explode(F.split('string', '[ \.,\\t]')).alias('word')) \
                .where(F.length('word') > 0) \
                .select('_id', '_key', F.trim(F.col('word')).alias('word'))
    
        self.total_stop_words = self.total_stop_words.select("word").rdd.flatMap(lambda x: x).collect()
        
        finished = total_corpus.filter(~total_corpus['word'].isin(self.total_stop_words))
        finished = finished.groupby("_id","_key").agg(
            F.concat_ws(",", F.collect_list(F.col('word'))).alias('keywords'),
            F.count(F.col('word')).alias('cnt')
            )
        
        hscd_df = finished.filter(F.col('_key') == 0)
        coid_df = finished.filter(F.col('_key') == 1)
        return hscd_df, coid_df
        
    def make_stopwords(self, sampled_corpus = None, tfidf_thresholds = 0.01, min_count = 10):
        sampled_corpus = sampled_corpus.select(['_id','string'])
        sampled_corpus = sampled_corpus \
                        .select('_id', F.explode(F.split('string', '[ \.,\\t]')).alias('word')) \
                        .where(F.length('word') > 0) \
                        .select('_id', F.trim(F.col('word')).alias('word'))

        def tf(df):
            w = Window.partitionBy(df['_id'])
            df = df.groupBy('_id', 'word').agg(
                        F.count('*').alias('n_w'),
                        F.sum(F.count('*')).over(w).alias('n_d'),
                        (F.count('*')/F.sum(F.count('*')).over(w)).alias('tf')
                    )
            return df
        
        def idf(df):
            w = Window.partitionBy('word')
            c_d = df.select('_id').distinct().count()
            df = df.groupBy('word', '_id').agg(
                    F.lit(c_d).alias('c_d'),
                    F.count('*').over(w).alias('i_d'),
                    F.log(F.lit(c_d)/F.count('*').over(w)).alias('idf')
                )
            return df

        tf_corpus = tf(sampled_corpus)
        idf_corpus = idf(sampled_corpus)
        tf_idf_corpus = tf_corpus.join(idf_corpus, ['_id', 'word'])\
                        .withColumn('tf_idf', F.col('tf') * F.col('idf'))\
                        .cache()
        
        min_count_stops = tf_corpus.groupBy('word').agg(F.sum(F.col('n_w')).alias('n_w'))
        
        tf_stops = tf_idf_corpus.filter(col("tf_idf") < tfidf_thresholds)
        min_count_stops = min_count_stops.filter(col("n_w") < min_count)

        total_stops = tf_stops.select(F.col('word')).union(min_count_stops.select(F.col('word'))).distinct()
        return total_stops

if __name__ == "__main__":
    # Input  : config file
    # Output : 결과 파일         config file path + '.extraction' (save in same path)
    #          메타 정보 딕셔너리  config file path + '.meta.dict'  (save in same path)
    #          메타 정보 텍스트    config file path + '.meta'       (save in same path)

    os.chdir("/IRIS/EMB_doc2vec/src")
    print(os.getcwd())
    
    config = configparser.ConfigParser()
    config.read("./conf/keyword_config.conf")
    st = time.time()
    ck = KeywordExtractor(config)
    hscd_df, coid_df = ck.make_keywords_df()

    # ck.save_keywords()
    # ck.run()
