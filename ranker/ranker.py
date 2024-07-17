from collections import defaultdict

import numpy as np
import torch
from summarizer.sbert import SBertSummarizer
from transformers import BertTokenizer

from db.DocumentEntry import DocumentEntry
from db.DocumentRepository import DocumentRepository
from ranker.QueryResult import QueryResult


class Ranker:

    def __init__(self):
        self.documentRepository = DocumentRepository()

        # Here we are using the Bert-Tokenizer to get a map from words to int and vice versa.
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.summarizer = SBertSummarizer('paraphrase-MiniLM-L6-v2')

        print("Loading Documents first...")
        self.all_docs = self.documentRepository.loadAllDocuments()
        print("All documents loaded.")

        print("Now we load TF and IDF")
        self.idf = self.documentRepository.load_idf()
        self.tf = self.documentRepository.load_tf()
        self.doc_lengths, self.max_term, self.avg_doc_len = self.documentRepository.load_tf_metadata()
        print("TF and IDF loaded.")

    def rank_query(self, query: str, documents, n=100):
        """
        Returns the top n documents of a given query.

        Parameters
        query: The input query of the user.
        n: The top n documents to return. Default 100.

        Returns
        The top n ranked documents for that query.
        """

        # 1: calculate BM25 and keep the ranking for the top n documents
        bm25_scores = self.rank_BM25v2(query, documents, n=n)

        # 2: rerank the top n documents again with a neural ranker
        neural_scores = bm25_scores  # TODO: df: Implement. This is for you Lilli and Lenard!

        # 3 (Optional): combining scores to get the final ranking.
        final_ranking = neural_scores  # TODO: df: Implement. This is only optional if we want to ensemble rankings. Otherwise we can just delete it

        return final_ranking

    def _calculate_tf_and_lengthsv2(self, tensor_docs):
        """
        Represents term frequency and length of documents.
        WARNING: only used for offline computation. Otherwise, BM25 would be too slow

        Results:
        term_frequency: Term frequency as a tensor for each token.
        doc_lengths: The length of all documents.
        max_term: The maximum supported token of the term_frequency tensor.
        """

        # mask for valid terms (non-padded elements)
        valid_mask = tensor_docs != -1

        # calculate document lengths
        doc_lengths = valid_mask.sum(dim=1).tolist()

        # create a tensor to hold term frequencies
        max_term = int(tensor_docs.max().item())
        term_frequency = torch.zeros((tensor_docs.size(0), max_term + 1), dtype=torch.int)

        # count term frequencies
        for doc_idx, doc in enumerate(tensor_docs):
            for term in doc[valid_mask[doc_idx]]:
                term_frequency[doc_idx, term] += 1

        return term_frequency, doc_lengths, max_term

    def _compute_idfv2(self, tensor_docs):
        """
        Computes IDF with fancy tensor magic.
        WARNING: only used for offline computation. Otherwise, BM25 would be too slow

        Parameter:
        tensor_docs nxm tensor: Contains n documents as its rows and a padded representation of the text as columns. m are the maximum number of tokens in 1 of these n documents.
        If a text of a document is too small, it gets padded up using -1.

        Returns:
        dict: tokens as integer keys and idf_scores as float values
        """

        N = tensor_docs.shape[0]  # Number of documents
        term_doc_count = defaultdict(int)

        for doc in tensor_docs:
            unique_terms = torch.unique(doc[doc != -1])
            for term in unique_terms:
                term_doc_count[term.item()] += 1

        unique_terms = list(term_doc_count.keys())
        idf_scores = [np.log(N / (term_doc_count[term] + 1)) for term in unique_terms]

        return dict(zip(unique_terms, idf_scores))

    def rank_BM25v2(self, query: str, documents: list[DocumentEntry], k=1.5, b=0.75, n=100):
        """
        Calculate the Okapi Best Model 25 scores for a set of documents given a query.
        The BM25 score for a document D given a query Q is calculated as:
        BM25(D, Q) = Σ [ IDF(q_i) * (f(q_i, D) * (k + 1)) / (f(q_i, D) + k * (1 - b + b * (|D| / avgdl))) ]

        Note: The BM25 score of a document is <= 0 but can be > 1.

        Where:
            - q_i: the i-th term in the query Q.
            - f(q_i, D): term frequency of q_i in document D.
            - |D|: length of document D.
            - avgdl: average document length in the corpus.
            - k: controls the term frequency saturation. Typical values range from 1.2 to 2.0.
            - b: controls the length normalization. Typical values range from 0.75 to 1.0.
            - IDF(q_i): inverse document frequency of the term q_i.

        Parameters:
            query (string): the query of the user.
            documents (list of DocumentEntry): list of all documents available
            k (float, optional): term frequency saturation parameter. Default is 1.5.
            b (float, optional): length normalization parameter. Default is 0.75.

        Returns:
            QueryResult: the results of the BM25 calculation.
        """

        # prep: the good thing is that this implementation uses precomputed tf and idf scores which makes BM25 faster
        number_docs = len(documents)
        tokenized_query = self.tokenizer.encode(query, add_special_tokens=False)

        # compute scores
        scores = torch.zeros(number_docs)
        for term in tokenized_query:
            if term in self.idf:
                idf_term = self.idf.get(term)
                for doc_idx in range(number_docs):
                    tf_term_doc = self.tf[doc_idx][term] if term <= self.max_term else torch.tensor([0])
                    doc_length = int(self.doc_lengths[doc_idx])
                    numerator = tf_term_doc * (k + 1)
                    denominator = tf_term_doc + k * (1 - b + b * doc_length / self.avg_doc_len)
                    scores[doc_idx] += idf_term * numerator / denominator

        # last step: sort them and only keep the top_n
        sorted_indices = torch.argsort(-scores)[:n]
        sorted_scores = scores[sorted_indices].tolist()
        sorted_docs = [documents[i] for i in sorted_indices]

        return QueryResult(query, sorted_docs, sorted_scores)

    def _compute_tf(self, token, doc_vec):
        return np.sum(doc_vec == token)

    def _compute_idf(self, documents: list[np.array]):
        N = len(documents)
        idf = {}
        for document in documents:
            for word in document:
                if word in idf:
                    idf[word] += 1
                else:
                    idf[word] = 1
        for word, count in idf.items():
            idf[word] = np.log(((N + 1) / (count + 0.5)) + 1)
        return idf

    def rank_BM25(self, tokenized_query: list[int], documents_vectors: dict[str, np.array], k=1.5, b=0.75):
        """
        df: deprecated. Only kept for old experiments

        Calculate the Okapi Best Model 25 scores for a set of documents given a query.
        The BM25 score for a document D given a query Q is calculated as:
        BM25(D, Q) = Σ [ IDF(q_i) * (f(q_i, D) * (k + 1)) / (f(q_i, D) + k * (1 - b + b * (|D| / avgdl))) ]

        Note: The BM25 score of a document is <= 0 but can be > 1.

        Where:
            - q_i: the i-th term in the query Q.
            - f(q_i, D): term frequency of q_i in document D.
            - |D|: length of document D.
            - avgdl: average document length in the corpus.
            - k: controls the term frequency saturation. Typical values range from 1.2 to 2.0.
            - b: controls the length normalization. Typical values range from 0.75 to 1.0.
            - IDF(q_i): inverse document frequency of the term q_i.

        Parameters:
            tokenized_query (list of int): list of tokens.
            documents_vectors (dict of (str, np.array)): contains for each url the encoded np.array
            k (float, optional): term frequency saturation parameter. Default is 1.5.
            b (float, optional): length normalization parameter. Default is 0.75.

        Returns:
            Dict[str, float]: a dictionary where keys are the urls and values are the corresponding BM25 scores.
        """
        # Compute the number of documents
        n = len(documents_vectors)

        # Can be precomputed ------
        # Compute average document length
        vectors_list = list(documents_vectors.values())
        avg_doc_len = np.mean([len(i) for i in vectors_list])

        # Compute IDF for all terms in the corpus
        idf = self._compute_idf(vectors_list)
        # --------

        scores = {}

        # Compute term frequency IF(q, d)
        for url, doc_vec in documents_vectors.items():
            score = 0
            doc_length = len(doc_vec)

            for token in tokenized_query:
                tf = self._compute_tf(token, doc_vec)
                numerator = (idf.get(token, np.log(
                    n + 1))  # assuming for a new word to have a high IDF-value because it's "rare".
                             * tf * (k - 1))  # frequency counting penalty
                denominator = tf + k * (1 - b + b * (doc_length / avg_doc_len))  # length norm
                score += numerator / denominator

            scores[url] = score

        return scores

    def _compute_score_for_token(self, token, doc_vec, idf, avg_doc_len, doc_length, k, b, n):
        """
        df: deprecated. Only kept for old experiments
        """

        tf = np.sum(doc_vec == token)
        numerator = idf.get(token, np.log(n + 1)) * tf * (
                k + 1)  # assuming for a new word to have a high IDF-value because it's "rare".
        denominator = tf + k * (1 - b + b * (doc_length / avg_doc_len))  # length norm
        return numerator / denominator

    def add_summary_description(self, final_ranking: QueryResult):
        """
        df: deprecated. Only kept for old experiments
        """

        documents = final_ranking.documents
        summaries = []
        for i, doc in enumerate(documents):
            text = doc.page_text
            if i >= 6:
                summary = text
            else:
                summary = self.summarizer(text, min_length=60, num_sentences=3, max_length=300)
            summaries.append(summary)
        final_ranking.summaries = summaries
        return final_ranking
