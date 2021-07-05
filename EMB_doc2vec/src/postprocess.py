from gensim.models import KeyedVectors
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time


def read_key_value_words(hscd_path, coid_path):
    def __file_reader(path):
        with open(path, "r") as file:
            while True:
                line = file.readline()
                if line == "":
                    break
                yield line.strip("\n")

    return __file_reader(hscd_path), __file_reader(coid_path)


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

    for _id, _hscode in enumerate(hscd_vectors):
        hscode_cos_mat = np.concatenate([_tmp[_id] for _tmp in total_cos_mat], axis=-1)
        hscode_coid_sorted_index = np.argsort(-hscode_cos_mat, axis=-1)
        print(hscode_coid_sorted_index)


class postprocess:
    def __init__(self, model, config):
        model_config = config["model"]
        self.model = model
        self.hscode_path = model_config["hscode_path"]
        self.coid_path = model_config["coid_path"]

    def run(self, hscd_vectors, coid_vectors, n_split=100):
        calculate_cosine(hscd_vectors, coid_vectors, n_split)


class post_doc2vec(postprocess):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.doctags, self.vectors = self.__check_d2v_index_and_values_order()
        self.hscd_tags, self.hscd_vectors, self.coid_tags, self.coid_vectors = self._split_vectors(self.hscode_path, self.coid_path)
        self.run(self.hscd_vectors, self.coid_vectors)

    def __check_d2v_index_and_values_order(self):
        doctags = self.model.docvecs.index2entity
        vectors = self.model.docvecs.vectors_docs
        for _index, _id in enumerate(doctags):
            assert np.array_equal(self.model.docvecs[_id], vectors[_index]), "Check Failed..!"
        return doctags, vectors

    def _split_vectors(self, hscode_path, coid_path):
        def _get_doctag_and_vector(model, gen_string):
            _doctags = model.docvecs.index2entity
            tags = [i.split("|")[0] for i in gen_string]
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
        self.run(self.hscd_vectors, self.coid_vectors)

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
        def _get_key_words_centroids(gen_string):
            tags = []
            vectors = []
            for _string in gen_string:
                key, words = _string.strip().split("|")
                words = words.split(",")
                centroid_vector = self.get_centroid(words)
                tags.append(key)
                vectors.append(centroid_vector)
            return tags, np.stack(vectors)

        gen_hscode_string, gen_coid_string = read_key_value_words(hscode_path, coid_path)
        hscode_tags, hscode_vectors = _get_key_words_centroids(gen_hscode_string)
        coid_tags, coid_vectors = _get_key_words_centroids(gen_coid_string)

        return hscode_tags, hscode_vectors, coid_tags, coid_vectors


if __name__ == "__main__":
    import configparser

    config = configparser.ConfigParser()
    config.read("./conf/post_w2v.conf")

    model = KeyedVectors.load_word2vec_format(config["model"]["model_path"], binary=True)
    post_processor = post_word2vec(model, config)

if __name__ == "__main__":
    import configparser

    config = configparser.ConfigParser()
    config.read("./conf/post_d2v.conf")

    model = Doc2Vec.load(config["model"]["model_path"])
    post_processor = post_doc2vec(model, config)
