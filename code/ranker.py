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
        self.number_of_documents += 1
        self.average_document_length = new_total_length / self.number_of_documents
    
    def bm25(self, query, topn=10, k1=1.2, b=0.75):
        """ Ranks documents against given query using BM25 method
        query: string containing the user query
        topn: int number of documents to be returned (default = 10)
        k1: int parameter to control
