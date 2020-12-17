# Methods to load and preprocess various documents

import re


def load_trek_2005_robust_document(document_path):
    """Returns an array of the individual documents

    :param document_path: the path to a TREK 2005 robust document txt file
    :return: array of strings
    """
    documents = []
    with open(document_path,"r") as f:
        for line in f:
            document = []
            if "<TEXT>" in line:   # Starting point to store following lines
                for line in f:
                    if "</TEXT>" in line: # Cutoff point to stop storing lines
                        documents.append(" ".join(document))
                        break
                    else:
                        document.append(line.strip())
    return documents

def process_document_w2v(document, stopwords=False, min_length=3):
    """Returns a processed text document for word2vec embedding input

    :param document: text document as a single string
    :return: array split by sentence contained individual words
    """
    document = document.lower()
    document = document.split(". ")
    for sentence in document:
        sentence = re.sub("r[^a-zA-Z]+", "", sentence)
    
    print(document) 
                
x = load_trek_2005_robust_document("/home/kjros2/query_expansion/data/trek_2005_robust/documents/19981217_APW_ENG")
process_document_w2v(x[0])
