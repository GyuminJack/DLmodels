from gensim.models import KeyedVectors
from gensim.models.doc2vec import Doc2Vec
import numpy as np
import time
from utils import *

def calculate_cosine(hscd_vectors, coid_vectors, n_split):
    def __split_indices(a, n):
        # Generator
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))

    if n_split > len(coid_vectors):
        n_split = 1

    split_indices = __split_indices(range(len(coid_vectors)), n_split)

    total_cos_mat = []
    for _id, _splitted_coid_index in enumerate(split_indices):
        cos_mat = cosine_similarity(hscd_vectors, coid_vectors[_splitted_coid_index])
        total_cos_mat.append(cos_mat)
    
    return total_cos_mat

class postprocess:
    def __init__(self, model, config):
        model_config = config["model"]
        self.model = model
        self.hscode_path = model_config["hscode_path"]
        self.coid_path = model_config["coid_path"]
        self.vector_save_path = model_config['vector_save_path']
        os.makedirs(self.vector_save_path, exist_ok = True)
        self.cos_mats = None
        self.hscd_tags, self.hscd_vectors = None, None
        self.coid_tags, self.coid_vectors = None, None
        
    def get_hscodes(self):
        return self.hscd_tags

    def get_coids(self):
        return self.coid_tags

    def _split_vectors(self, hscode_path, coid_path):
        raise NotImplementedError("_split_vectors method")

    def make_cosmat(self, hscd_vectors, coid_vectors, n_split=100):
        return calculate_cosine(hscd_vectors, coid_vectors, n_split)

    def save_files(self):
        writer(self.hscd_tags, os.path.join(self.vector_save_path, 'hscd.tags'))
        writer(self.coid_tags, os.path.join(self.vector_save_path, 'coid.tags'))
        np.save(os.path.join(self.vector_save_path, 'hscd.npy'), self.hscd_vectors)
        np.save(os.path.join(self.vector_save_path, 'coid.npy'), self.coid_vectors)

    def save_bulk_cosmat(self, cos_mat:list, hscd_tags, coid_tags):
        def save(tag, rank_vector):
            # 수정 필요 ***
            # 일단 파일로 떨구고 그 다음에 다른 모듈에서 불러다가 서비스를 위해 상위 몇만개 추출하는 시나리오로 예상
            print(tag, rank_vector)

        # HSCODE 기준 유사도 순위 생성 (max_rank = len(coid))
        for _id, _tag in enumerate(hscd_tags):
            hscode_cos_mat = np.concatenate([_tmp[_id] for _tmp in cos_mat], axis=-1)
            hscode_coid_sorted_index = np.argsort(-hscode_cos_mat, axis=-1) # HSCODE(1) - COID (max_rank = 6000000)
            save(_tag, hscode_coid_sorted_index)
        
        # COID 기준 유사도 순위 생성 (max_rank = len(hscode))
        start = 0
        for _id, sub_cos_mat in enumerate(cos_mat):
            end = start + sub_cos_mat.shape[-1]
            _tag = coid_tags[start : end]
            print(sub_cos_mat.T)
            coid_hscode_sorted_index = np.argsort(-sub_cos_mat.T, axis=-1) # sub_coid_range * 6000
            for _t, _r in zip(_tag, coid_hscode_sorted_index):
                save(_t, _r)
            start = end

    def run(self):
        self.cos_mats = self.make_cosmat(self.hscd_vectors, self.coid_vectors)
        self.save_bulk_cosmat(self.cos_mats, self.hscd_tags, self.coid_tags)

class post_doc2vec(postprocess):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.__check_d2v_index_and_values_order()
        self.hscd_tags, self.hscd_vectors, self.coid_tags, self.coid_vectors = self._split_vectors(self.hscode_path, self.coid_path)
        self.save_files()
        self.run()

    def __check_d2v_index_and_values_order(self):
        doctags = self.model.docvecs.index2entity
        vectors = self.model.docvecs.vectors_docs
        # doctag의 순서와 벡터의 순서가 서로 같은지 검증하는 과정
        for _index, _id in enumerate(doctags):
            assert np.array_equal(self.model.docvecs[_id], vectors[_index]), "Check Failed..!"
    
    def _split_vectors(self, hscode_path, coid_path):
        # 바뀔수도..?
        def __make_tag_with_string(string):
            tag, _ = string.split("|")
            return tag

        def _get_doctag_and_vector(model, gen_string):
            _doctags = model.docvecs.index2entity
            tags = [__make_tag_with_string(i) for i in gen_string]
            _doctag_index = [_doctags.index(tag) for tag in tags]
            _vectors = model.docvecs.vectors_docs[_doctag_index].reshape(-1, model.docvecs.vector_size)
            return tags, _vectors

        gen_hscode_string, gen_coid_string = read_key_value_words(hscode_path, coid_path)
        hscode_tags, hscode_vectors = _get_doctag_and_vector(self.model, gen_hscode_string)
        coid_tags, coid_vectors = _get_doctag_and_vector(self.model, gen_coid_string)

        return hscode_tags, hscode_vectors, coid_tags, coid_vectors


class post_word2vec(postprocess):
    def __init__(self, word2vec, config):
        super().__init__(model, config)
        self.hscd_tags, self.hscd_vectors, self.coid_tags, self.coid_vectors = self._split_vectors(self.hscode_path, self.coid_path)
        self.save_files()
        self.run()

    def get_centroid(self, words_list):
        np_words = []
        for word in words_list:
            try:
                np_words.append(self.model[word])
            except Exception as e:
                pass

        if len(np_words) == 0:
            return np.zeros(self.model.vector_size)
        else:
            return np.mean(np.stack(np_words), axis=0)

    def _split_vectors(self, hscode_path, coid_path):
        def __make_key_and_words(string):
            key, words = __make_key_and_words
            words = words.split(",")
            return key, words

        def _get_key_words_centroids(gen_string):
            tags = []
            vectors = []
            for _string in gen_string:
                _tag, _words = __make_key_and_words(_string)
                centroid_vector = self.get_centroid(_words)
                tags.append(_tag)
                vectors.append(centroid_vector)
            return tags, np.stack(vectors)

        gen_hscode_string, gen_coid_string = read_key_value_words(hscode_path, coid_path)
        hscode_tags, hscode_vectors = _get_key_words_centroids(gen_hscode_string)
        coid_tags, coid_vectors = _get_key_words_centroids(gen_coid_string)

        return hscode_tags, hscode_vectors, coid_tags, coid_vectors


if __name__ == "__main__":
    import configparser
    import sys
    import os

    mode = sys.argv[1]
    config = configparser.ConfigParser()
    root_path = "/home/jack/dlmodels/EMB_doc2vec/src"
    os.chdir(root_path)

    if mode == 'w2v':
        config.read("./conf/post_w2v.conf")
        model = KeyedVectors.load_word2vec_format(config["model"]["model_path"], binary=True)
        post_processor = post_word2vec(model, config)
    elif mode == 'd2v':
        config.read("./conf/post_d2v.conf")
        model = Doc2Vec.load(config["model"]["model_path"])
        post_processor = post_doc2vec(model, config)

