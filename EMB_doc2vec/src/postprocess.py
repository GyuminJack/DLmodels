from gensim.models import KeyedVectors
from gensim.models.doc2vec import Doc2Vec
import numpy as np
import time
from utils import *
from numba import njit
from multiprocessing import Pool
import gc
import sys

def calculate_cosine(hscd_vectors, coid_vectors, n_split):
    def __split_indices(a, n):
        # Generator
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))

    if n_split > len(coid_vectors):
        n_split = 10

    split_indices = __split_indices(range(len(coid_vectors)), n_split)

    total_cos_mat = [None] * n_split
    for _id, _splitted_coid_index in enumerate(split_indices):
        total_cos_mat[_id] = cosine_similarity(hscd_vectors, coid_vectors[_splitted_coid_index]).astype(np.float16)
    
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

    @time_printer(True)
    def make_cosmat(self, hscd_vectors, coid_vectors, n_split=100):
        # return size : [[hscd * coid/n_split] * n_split]
        return calculate_cosine(hscd_vectors, coid_vectors, n_split)

    def save_files(self):
        writer(self.hscd_tags, os.path.join(self.vector_save_path, 'hscd.tags'))
        writer(self.coid_tags, os.path.join(self.vector_save_path, 'coid.tags'))
        np.save(os.path.join(self.vector_save_path, 'hscd.npy'), self.hscd_vectors)
        np.save(os.path.join(self.vector_save_path, 'coid.npy'), self.coid_vectors)
    
    def argsort_by_hscd_vector(self, hscd_id):
        # return size : 1 * coid length
        hscode_cos_vector = np.concatenate([each_cos[hscd_id] for each_cos in self.cos_mats], axis=-1)
        return -hscode_cos_vector.argsort(axis=-1)

    def argsort_by_coid_matrix(self, sub_cosine_matrix):
        # return size : sub coid vector * hscd length
        return -sub_cosine_matrix.T.argsort(axis=-1)

    @time_printer(True)
    def sort_and_save_cosmat(self, cos_mat:list, hscd_tags, coid_tags):
        def save(tag, rank_vector):
            # 수정 필요 ***
            # 일단 파일로 떨구고 그 다음에 다른 모듈에서 불러다가 서비스를 위해 상위 몇만개 추출하는 시나리오로 예상
            pass

        # HSCODE 기준 유사도 순위 생성 (max_rank = len(coid))
        # BEST : 2s
        for _id, _tag in enumerate(hscd_tags):
            hscode_coid_sorted_index = self.argsort_by_hscd_vector(_id)
            save(_tag, hscode_coid_sorted_index)
        
        # COID 기준 유사도 순위 생성 (max_rank = len(hscode))
        start = 0
        for sub_cos_mat in cos_mat:
            end = start + sub_cos_mat.shape[-1]
            _tag = coid_tags[start : end] # Just find tag name for save
            start = end

            coid_hscode_sorted_index = self.argsort_by_coid_matrix(sub_cos_mat)
            for _t, _r in zip(_tag, coid_hscode_sorted_index):
                save(_t, _r)
    
    @time_printer(True)
    def run(self):
        self.cos_mats = self.make_cosmat(self.hscd_vectors, self.coid_vectors)
        self.model, self.hscd_vectors, self.coid_vectors = None, None, None
        self.sort_and_save_cosmat(self.cos_mats, self.hscd_tags, self.coid_tags)

class post_doc2vec(postprocess):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.__check_d2v_index_and_values_order()
        self.hscd_tags, self.hscd_vectors, self.coid_tags, self.coid_vectors = self._split_vectors(self.hscode_path, self.coid_path)
        self.save_files()
        self.run()

    @time_printer(True)
    def __check_d2v_index_and_values_order(self):
        doctags = self.model.docvecs.index2entity
        vectors = self.model.docvecs.vectors_docs
        # doctag의 순서와 벡터의 순서가 서로 같은지 검증하는 과정
        for _index, _id in enumerate(doctags):
            assert np.array_equal(self.model.docvecs[_id], vectors[_index]), "Check Failed..!"
    
    @time_printer(True)
    def _split_vectors(self, hscode_path, coid_path):
        # 바뀔수도..?
        def __make_tag_with_string(string):
            tag, _ = string.split("|")
            return tag

        def _get_doctag_and_vector(model, gen_string):
            _doctags = model.docvecs.index2entity
            tags = [__make_tag_with_string(i) for i in gen_string]
            _doctag_index = [_doctags.index(tag) for tag in tags]
            _vectors = model.docvecs.vectors_docs[_doctag_index].reshape(-1, model.docvecs.vector_size).astype(np.float16)
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
            return np.zeros(self.model.vector_size).astype(np.float16)
        else:
            return np.mean(np.stack(np_words), axis=0).astype(np.float16)

    def _split_vectors(self, hscode_path, coid_path):
        def __make_key_and_words(string):
            key, words = string.split("|")
            words = words.split(",")
            return key, words

        @time_printer(True)
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


        # argsort Ideas
        
        # Too Slow
        # hscode_cos_mat = np.concatenate(cos_mat, axis=-1)
        # hscode_coid_sorted_index = np.argpartition(hscode_cos_mat, 
        #         range(hscode_cos_mat.shape[1]), axis=-1)[:,:hscode_cos_mat.shape[1]]
        # exit()

        # numba : 4s
        # for _id, _tag in enumerate(hscd_tags):
        #     hscode_cos_vector = np.concatenate([each_cos[_id] for each_cos in cos_mat], axis=-1)
        #     hscode_coid_sorted_index = postprocess.nb_sort(-hscode_cos_vector) # HSCODE(1) - COID (max_rank = 6000000)
        #     # hscode_coid_sorted_index = np.argpartition(hscode_cos_vector, 
        #     #     range(len(hscode_cos_vector)), axis=-1)[:len(hscode_cos_vector)]
        #     save(_tag, hscode_coid_sorted_index)

        # save time : 1min 10s
        # with open("cosine_test_npy_hscd6k.txt", "ab") as f:
        #     for _id, _tag in enumerate(hscd_tags):
        #         hscode_cos_vector = np.concatenate([each_cos[_id] for each_cos in cos_mat], axis=-1)
        #         np.savetxt(f, hscode_cos_vector)
        #         f.write(b"\n")
        #         # hscode_coid_sorted_index = np.argsort(-hscode_cos_vector, axis=-1) # HSCODE(1) - COID (max_rank = 6000000)
        #         # hscode_coid_sorted_index = np.argpartition(hscode_cos_vector, 
        #         #     range(len(hscode_cos_vector)), axis=-1)[:len(hscode_cos_vector)]
        #         # save(_tag, hscode_coid_sorted_index)