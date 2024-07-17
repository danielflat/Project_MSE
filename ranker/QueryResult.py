from db.DocumentEntry import DocumentEntry


class QueryResult:
    """
    This is the datastructure for the final representation of the query result which gets forwarded to the HTML.
    """
    def __init__(self, query: str, documents: list[DocumentEntry], scores: list[float]):
        """
        Parameters:
            query(str): The query of the user
            documents(list[DocumentEntry]): The suggested documents for the query
            scores(list[float]): The corresponding matching score for each document
            summaries(list[string]): The summary description for each document. Is only added after ranking in ranker.add_summary_description()

            Note: The ith document has the ith score and the ith summary text!
        """
        self.query = query
        self.documents = documents
        self.scores = scores
        self.summaries = None

