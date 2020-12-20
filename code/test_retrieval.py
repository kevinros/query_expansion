import preprocess
import ranker as r
import os

def setup_ranker(path_to_documents):
    ranker = r.Ranker()
    for filename in os.listdir(path_to_documents):
        docs = preprocess.load_trek_2005_robust_document(path_to_documents + filename)
        for doc in docs:
            processed_doc = preprocess.process_document(docs[doc])
            doc_by_word = [word for sentence in processed_doc for word in sentence]
            ranker.add_document(processed_doc, doc)
    return ranker

path_to_documents = "/home/kjros2/query_expansion/data/trek_2005_robust/documents/"

ranker = setup_ranker(path_to_documents)
print(ranker.bm25(["president", "europe"]))
