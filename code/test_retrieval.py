import preprocess
import ranker as r
import os
import pickle

def setup_ranker(path_to_documents, ranker_name):
    ranker = r.Ranker()
    for filename in os.listdir(path_to_documents):
        docs = preprocess.load_trek_2005_robust_document(path_to_documents + filename)
        for doc in docs:
            processed_doc = preprocess.process_document(docs[doc])
            doc_by_word = [word for sentence in processed_doc for word in sentence]
            ranker.add_document(processed_doc, doc)
    pickle.dump(ranker, open("../rankers/" + ranker_name, "wb"))
    return ranker

def load_ranker(path_to_ranker):
    return pickle.load(open(path_to_ranker, "rb"))

def load_trek_robust_topics(path_to_topics):
    topics = {}
    with open(path_to_topics) as f:
        for line in f:
            if "<num>" in line:
                topics[line[14:].strip()] = f.next()[8:].strip()
    return topics
                
def load_trek_robust_scores(path_to_scores):
    scores = {}
    with open(path_to_scores) as f:
        for line in f:
            split_line = line.strip().split()
            if split_line[0] in scores:
                scores[split_line[0]].append((split_line[3], split_line[2]))
            else:
                scores[split_line[0]] = [(split_line[3], split_line[2])]
    return scores


#path_to_documents = "/home/kjros2/query_expansion/data/trek_2005_robust/documents/"
#ranker = setup_ranker(path_to_documents, "small_trek.p")

#ranker = load_ranker("../rankers/small_trek.p")
#print('loaded')
#print(ranker.bm25(["president", "europe"]))

path_to_topics = "/home/kjros2/query_expansion/data/trek_2005_robust/metadata/05.50.topics.txt"
path_to_scores = "/home/kjros2/query_expansion/data/trek_2005_robust/metadata/TREC2005.qrels.txt"
x = load_trek_robust_scores(path_to_scores)
print(x['303'])
