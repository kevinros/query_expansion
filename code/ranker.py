import math

class Ranker(object):
    """
    Ranker class for ranking documents
    """
    
    def __init__(self):
        """
        Initializes the ranker
        """
        self.inverted_index = {}
        self.number_of_documents = 0
        self.average_document_length = 0
        # need to store relevance scores for documents!

        self.document_metadata = {} # Contains document length
        
    def add_document(self, document_text, document_id):
        """ Adds a document to the inverted index, and updates the global variables
        document_text: list of words in document
        document_id: string the id of the document (should match the relevance rankings)

        returns: nothing
        """
        for sentence in document_text:
            for word in sentence:
                if word in self.inverted_index:
                    if document_id in self.inverted_index[word]:
                        self.inverted_index[word][document_id] += 1
                    else:
                        self.inverted_index[word][document_id] = 1
                else:
                    self.inverted_index[word] = {}
                    self.inverted_index[word][document_id] = 1

        new_total_length = (self.average_document_length * self.number_of_documents) + len(document_text)
        self.document_metadata[document_id] = {}
        self.document_metadata[document_id]['length'] = len(document_text)
        self.number_of_documents += 1
        self.average_document_length = new_total_length / self.number_of_documents
    
    def bm25(self, query, topn=10, k1=1.2, b=0.75):
        """ Ranks documents against given query using BM25 method
        query: string containing the user query
        topn: int number of documents to be returned (default = 10)
        k1: int parameter to control
          
        returns: list of topn most similar documents
        """

        rankings = []
        for doc_id in self.document_metadata.keys():
            doc_score = 0
            for word in query:
                tf = self.TF_score_helper(word, doc_id)
                idf = self.IDF_score_helper(word)
                numerator = tf * (k1 + 1)
                denominator = tf + (k1 * (1 - b + (b * (self.document_metadata[doc_id]['length'] / self.average_document_length))))
                doc_score += idf * (numerator / denominator)
            rankings.append((doc_score,doc_id))
        rankings = sorted(rankings, key=lambda x: x[0], reverse=True)
        return rankings[0:topn]
  

    def TF_score_helper(self, keyword, doc_id):
        """Calculates the TF for BM25
        keyword: string word in query
        doc_id: string current document being scored                 
        returns: int TF score
        """

        try:
            tf = self.inverted_index[keyword][doc_id]
        except:
            tf = 0
        return tf

    def IDF_score_helper(self, keyword):
        """Calculates IDF for BM25
        keyword: string word in query
        returns: float IDF score
        """
        N = self.number_of_documents
        docs_containing_keyword = 0
        if keyword in self.inverted_index:
            docs_containing_keyword = len(self.inverted_index[keyword])
        numerator = N - docs_containing_keyword + 0.5
        denominator = docs_containing_keyword + 0.5
        return max(0, math.log((numerator / denominator) + 1))
