from sklearn.metrics.pairwise import cosine_similarity

class post_doc2vec:
    def __init__(self, doc2vec, hscode_tags):
        self.model = doc2vec
        self.doctags, self.vectors = self._check_d2v_vectors()
        self.hscd_tags, self.hscd_vectors, self.coid_tags, self.coid_vectors = _split_vectors(hscode_tags)

    def _check_d2v_vectors(self):
        doctags = self.model.docvecs.index2entity
        vectors = self.model.docvecs.vectors_docs
        for _index, _id in enumerate(doctags):
            assert np.array_equal(self.model.docvecs[_id], vectors[_index]), "Check Failed..!"
        return doctags, vectors

    def _split_vectors(self):
        def _get_doctag_and_vector(model, tags):        
            _doctags = model.docvecs.index2entity
            _doctag_index = [_doctags.index(tag) for tag in tags]
            _vectors = model.docvecs.vectors_docs[_doctag_index]
            return tags, _vectors

        hscode_tags, hscode_vectors = _get_doctag_and_vector(self.model, self.hscode_tags)
        coid_tags = [_tag for _tag in model.docvecs.index2entity if _tag not in hscode_tags]
        coid_tags, coid_vectors = _get_doctag_and_vector(model, coid_tags)
        
        return hscode_tags, hscode_vectors, coid_tags, coid_vectors

    def _calculate_cosine(self):
        def __split_indices(a, n):
            # Generator
            k, m = divmod(len(a), n)
            return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

        split_indices = __split_indices(range(len(self.coid_vectors)), 100)
        for _splitted_coid_index in split_indices:
            cos_mat = cosine_similarity(self.hscd_vectors, self.coid_vectors[_splitted_coid_index])
            # too Slow.
            ranks = np.argsort(cos_mat, axis=-1)
            

