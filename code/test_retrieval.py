import preprocess
import ranker as r
import os
import pickle
import math

def setup_ranker(path_to_documents, ranker_name):
    """
    Creates, saves, and returns ranker based on provided documents
    path_to_documents: string path to text documents
    ranker_name: string name to save ranker as

    returns: Ranker() class object
    """
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
    """
    Loads the trek robust topics from provided path
    path_to_topics: string path to topics

    returns: dict of topics[id] = topic
    """
    topics = {}
    with open(path_to_topics) as f:
        for line in f:
            if "<num>" in line:
                topics[line[14:].strip()] = f.next()[8:].strip()
    return topics
                
def load_trek_robust_scores(path_to_scores):
    """
    Loads the trek robust document scores per topic from provided path
    path_top_scores: string path to document scores
 
    returns: dict of scores[topic_id] = list of (score, doc_id)
    """
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

ranker = load_ranker("../rankers/small_trek.p")
print('loaded')

path_to_topics = "/home/kjros2/query_expansion/data/trek_2005_robust/metadata/05.50.topics.txt"
path_to_scores = "/home/kjros2/query_expansion/data/trek_2005_robust/metadata/TREC2005.qrels.txt"
all_scores = load_trek_robust_scores(path_to_scores)
all_topics = load_trek_robust_topics(path_to_topics)


def returned_doc_scores(topic_id, returned_docs, ground_truth_scores):
    """
    Given a list of returned documents from a search, maps the documents to the relevance scores
    topic_id: string id of the topic
    returned_docs: list of returned docs from the topic query
    ground_truth_scores: dict of topic scores for each document

    returns: list of doc relevance scores in order the documents appeared
    """
    topic_scores = ground_truth_scores[topic_id]
    doc_scores = []
    for returned_doc in returned_docs:
        in_topics = False
        for gt_doc in topic_scores:
            if gt_doc[1] == returned_doc[1]:
                doc_scores.append(int(gt_doc[0]))
                in_topics = True
                break
        if not in_topics:
            doc_scores.append(0)
    return doc_scores


def score_ndcg(scores):
    """
    Calculates NDCG given a list of scores
    scores: list of document scores

    returns: int ndcg
    """
    dcg = 0
    for i,s in enumerate(scores):
        dcg += s / math.log(i+2,2)
    sorted_scores = sorted(scores, key=lambda x: x, reverse=True)
    idcg = 0
    for i,s in enumerate(sorted_scores):
        idcg += s / math.log(i+2,2)
    if idcg == 0:
        return 0
    else:
        return dcg / idcg



for topic in all_topics.keys():
    returned_docs = ranker.bm25(all_topics[topic].lower().split(" "))
    returned_scores = returned_doc_scores(topic, returned_docs, all_scores)
    print(all_topics[topic])
    print(returned_scores)
    print(score_ndcg(returned_scores))
