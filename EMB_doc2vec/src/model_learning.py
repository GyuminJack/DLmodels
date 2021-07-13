from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class d2v_Inputs(object):
    def __init__(self, filename):
        self.filename = filename
    def __iter__(self):
        for line in (open(self.filename)):
            k, v = line.split("|")
            yield TaggedDocument(v.split(","), [k])

class w2v_Inputs(object):
    def __init__(self, filename):
        self.filename = filename
    def __iter__(self):
        for line in (open(self.filename)):
            _, v = line.split("|")
            yield v.split(",")

def run_d2v(corpus_path):
    tagged_data = []
    data = d2v_Inputs(corpus_path)
   
    max_epochs = 100
    vec_size = 300
    alpha = 0.025

    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha, 
                    min_alpha=0.00025,
                    min_count=1,
                    dm =1)
    
    model.build_vocab(data)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch), end="\r")

        model.train(data,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)

        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    model.save("/home/jack/dlmodels/EMB_doc2vec/models/d2v.0715.model")

def run_w2v(corpus_path):
    tagged_data = []
    data = w2v_Inputs(corpus_path)
       
    max_epochs = 100
    vec_size = 300
    alpha = 0.025

    model = Word2Vec(size=vec_size,
                    alpha=alpha, 
                    min_alpha=0.00025,
                    min_count=1
                    )
    
    model.build_vocab(data)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch), end="\r")

        model.train(data,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)

        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    model.save("/home/jack/dlmodels/EMB_doc2vec/models/w2v.0715.model")
    
def init_vectors(pretrained_model, real_model):
    pass

if __name__ == "__main__":
    corpus_path = "/home/jack/dlmodels/EMB_doc2vec/data/WikiQA-train.txt.extraction"
    # run_d2v(corpus_path)
    run_w2v(corpus_path)