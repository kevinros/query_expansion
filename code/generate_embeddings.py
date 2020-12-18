import gensim
from gensim.models import KeyedVectors
import preprocess
import os

path_to_documents = "/home/kjros2/query_expansion/data/trek_2005_robust/documents/"


def trek2corpus(trek_path):
    corpus = []
    for filename in os.listdir(trek_path):
        docs = preprocess.load_trek_2005_robust_document(trek_path + filename)
        for doc in docs:
            processed_doc = preprocess.process_document(doc)
            corpus += processed_doc   
    return corpus

corpus = trek2corpus(path_to_documents)
#print('corpus made')


def train_embedding(corpus,corpus_name,dimension,window,min_count, workers, sg,iterations):
    model = gensim.models.Word2Vec(corpus, size=dimension, window=window, min_count=min_count, workers=workers, sg=sg, iter=iterations)
    model.wv.save_word2vec_format('../models/' + corpus_name + '_' + str(dimension) + '_' + str(window) + '_' + str(min_count) + '.txt', binary=False)
       

train_embedding(corpus,"test",100,10,10,5,0,10)

def test_embedding(path_to_model,words):
    model = KeyedVectors.load_word2vec_format(path_to_model,binary=False)
    #print(model.wv.vocab)
    for word in words:
        try:
            print(word)
            print(model.wv.most_similar(word, topn=10))
        except:
            print('Word not in vocabulary')

test_embedding('../models/test_100_10_10.txt',['kids','golf','and','man','woman'])
                                                                                                                                                                                                            
