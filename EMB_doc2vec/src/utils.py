import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
    
    return total_cos_mat


def get_most_similar_coids(hscode_vector, coid_vectors, top_n = 20):
    hscode_vector = hscode_vector.reshape(-1, hscode_vector.shape[-1])
    cos_mat = cosine_similarity(hscode_vector, coid_vectors)
    return np.argpartition(-cos_mat, top_n)
    

class tag_vector_storage:
    def __init__(self, hscd_tag_path, hscd_vector_path, coid_tag_path, coid_vector_path):
        self.hscd_tag = reader(hscd_tag_path)
        self.coid_tag = reader(coid_tag_path)
        self.hscd_vector = np.load(hscd_vector_path)
        self.coid_vector = np.load(coid_vector_path)
        assert len(self.hscd_tag) == len(self.hscd_vector), f"Size Mismatched, HSCD (tag = {len(self.hscd_tag)}, vec = {len(self.hscd_vector)})"
        assert len(self.coid_tag) == len(self.coid_vector), f"Size Mismatched, COID (tag = {len(self.coid_tag)}, vec = {len(self.coid_vector)})"

    def recommand_coids(self, hscodes, target_coids = None, top_n = 20):
        hscode_vector = self.hscd_vector[[self.hscd_tag.index(hscode) for hscode in hscodes]]
        hscode_vector = hscode_vector.reshape(-1, hscode_vector.shape[-1])
        
        if target_coids is not None:
            target_coid_index = [self.coid_tag.index(coid) for coid in target_coids]
            target_coid_tag = [self.coid_tag[i] for i in target_coid_index]
            target_coid_vector = self.coid_vector[target_coid_index]
        else:
            target_coid_tag = self.coid_tag
            target_coid_vector =  self.coid_vector

        top_n = min(len(target_coid_vector), top_n)
        
        cos_mat = cosine_similarity(hscode_vector, target_coid_vector)
        indices = np.argpartition(-cos_mat, range(top_n), axis=-1)[:,:top_n]

        ret = []
        for _ln, each_index in enumerate(indices):
            each = [(target_coid_tag[_index], cos_mat[_ln, _index]) for _index in each_index]
            ret.append(each)
        return ret
    


if __name__ == '__main__':
    hscd_tag_path = "/home/jack/dlmodels/EMB_doc2vec/src/tmp_save/d2v_vec/hscd.tags"
    coid_tag_path = "/home/jack/dlmodels/EMB_doc2vec/src/tmp_save/d2v_vec/coid.tags"
    hscd_vec_path = "/home/jack/dlmodels/EMB_doc2vec/src/tmp_save/d2v_vec/hscd.npy"
    coid_vec_path = "/home/jack/dlmodels/EMB_doc2vec/src/tmp_save/d2v_vec/coid.npy"

    tv_st = tag_vector_storage(hscd_tag_path, hscd_vec_path, coid_tag_path, coid_vec_path)
    print(tv_st.recommand_coids(['00100'], top_n = 3))