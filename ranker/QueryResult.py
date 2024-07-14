from db.DocumentEntry import DocumentEntry


class QueryResult:

    def __init__(self, query: str, documents: list[DocumentEntry], scores: list[float]):
        self.query = query
        self.documents = documents
        self.scores = scores
        # self.ranked_documents = self._s   # TODO: Implement sorted

    # def __repr__(self):
    #     return f"Query: {self.query}, Score:{self.score}, Url:{self.document.url}"
