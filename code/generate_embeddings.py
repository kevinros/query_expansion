import gensim
from gensim.models import KeyedVectors
import preprocess
import os


def trek2corpus(trek_path):
    """ Creates the word2vec training corpus from a directory of TREK documents
    
    trek_path: string path to trek directory
    returns: list of sentences
    """
    corpus = []
    for filename in os.listdir(trek_path):
        docs = preprocess.load_trek_2005_robust_document(trek_path + filename)
        for doc in docs.keys():
            processed_doc = preprocess.process_document(docs[doc]) 
            corpus += processed_doc   
    return corpus


def train_embedding(corpus,corpus_name,dimension,window,min_count, workers, sg,iterations):
    """ Trains a word2vec model and saves it to the models directory

    corpus: A list of lists of sentences, each word as an element
    corpus_name: String name of the corpus, for naming the file
    dimension: int indicating the dimension of the produced vectors
    window: int the sliding window size
    min_count: int the minimum count of a word for it to be included
    workers: int the number of workers
    sg: 1 is skip_gram, 0 is CBOW
    iterations: int how many times to iterate over the training corpus

    returns: nothing, just writes the model to the model directory
   """
    model = gensim.models.Word2Vec(corpus, size=dimension, window=window, min_count=min_count, workers=workers, sg=sg, iter=iterations)
    model.wv.save_word2vec_format('../models/' + corpus_name + '_' + str(dimension) + '_' + str(window) + '_' + str(min_count) + '.txt', binary=False)
       

def test_embedding(path_to_model,words):
    """ Takes a path to word2vec model, and an array of words, 
        and returns the ten most similar words to each word provided
    path_to_model: path to word2vec model (assumes txt)
    words: an array of words to view the most similar words of
    returns: nothing, just prints the most similar words
    """ 
    model = KeyedVectors.load_word2vec_format(path_to_model,binary=False)
    for word in words:
        try:
            print(word)
            print(model.wv.most_similar(word, topn=10))
        except:
            print('Word not in vocabulary')



path_to_documents = "/home/kjros2/query_expansion/data/trek_2005_robust/documents/"
#corpus = trek2corpus(path_to_documents)
print('corpus made')
#train_embedding(corpus,"full",100,10,10,5,0,100) 
test_embedding('../models/test_100_10_10.txt',['kids','golf','and','man','woman'])
test_embedding('../models/full_100_10_10.txt',['kids','golf','and','man','woman'])                                                                                                                                                                                                             
