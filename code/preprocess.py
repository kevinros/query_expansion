# Methods to load and preprocess various documents

import re


def load_trek_2005_robust_document(document_path):
    """Returns an id-text dict of the individual documents

    :param document_path: the path to a TREK 2005 robust document txt file
    :return: dictionary of [id] = doc_text
    """
    documents = {}
    with open(document_path,"r") as f:
        for line in f:
            document = []
            if "<DOCNO>" in line:    # Get document id number
                doc_id = line[7:-9].strip() 
            
            if "<TEXT>" in line:   # Starting point to store following lines
                for line in f:
                    if "</TEXT>" in line: # Cutoff point to stop storing lines
                        if document and doc_id:
                            documents[doc_id] = " ".join(document)
                        else:
                            print('Could not find id or text for:')
                            print(doc_id,document)
                        break
                    else:
                        document.append(line.strip())
    return documents

def process_document(document, stopwords=False, min_length=2):
    """Returns a processed text document

    :param document: text document as a single string
    :return: array split by sentence contained individual words
    """
    
    # STOPWORDS?
    # STEMMING?
    #print(document)
    document = document.lower()
    document = document.split(". ")
    new_document = []
    for sentence in document:
        sentence = re.sub("[^a-zA-Z]+", " ", sentence).split(" ")
        
        sentence = [x for x in sentence if len(x) > min_length]
        if sentence:
            new_document.append(sentence)
    return new_document

