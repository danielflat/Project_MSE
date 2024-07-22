import heapq
import os
import time
from collections import defaultdict

import numpy as np
from IPython.display import display
from summarizer.sbert import SBertSummarizer

from db.DocumentEntry import DocumentEntry
from db.DocumentRepository import DocumentRepository
from ranker.QueryResult import QueryResult
from ranker.TextEmbeddings import *
from utils.directoryutil import get_path


class Ranker:

    def __init__(self):
        self.documentRepository = DocumentRepository()

        # Here we are using the Bert-Tokenizer to get a map from words to int and vice versa.
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.summarizer = SBertSummarizer('paraphrase-MiniLM-L6-v2')

        # Here we load the trained model for reranking

        self.max_length = 512  # Maximum length of a document
        self.embedding_dim = 128
        self.vocab_size = self.tokenizer.vocab_size
        print(f"Vocab size: {self.vocab_size}")

        print("Loading Documents first...")
        self.all_docs = self.documentRepository.loadAllDocuments()
        print("All documents loaded.")

        print("Now we load TF and IDF")
        self.idf = self.documentRepository.load_idf()
        self.tf = self.documentRepository.load_tf()
        self.doc_lengths, self.max_term, self.avg_doc_len = self.documentRepository.load_tf_metadata()
        print("TF and IDF loaded.")

        try:
            print("Loading trained model...")
            self.trained_ranker = self._load_trained_model()
            print("Trained model loaded.")
        except FileNotFoundError:
            print("No trained model found. Please train the model first.")

    def rank_query(self, query: str, documents, n=100):
        """
        Returns the top n documents of a given query.

        Parameters
        query: The input query of the user.
        n: The top n documents to return. Default 100.

        Returns
        The top n ranked documents for that query.
        """

        # 1: calculate BM25 and keep the ranking for the top 200 documents
        top_bm25_number = n   # Set it either to 'n' or to a number between 100 and 200
        print(f"top_bm25_number: {top_bm25_number}")
        bm25_scores = self.rank_BM25v2(query, documents, n=top_bm25_number)

        # 2 (Optional): diversity BM25 list to get more diverse documents. Keep the top n documents. Otherwise keep top 100 from BM25
        l = 1   # 0 <= l <= 1: controls the relevance rate. If l = 1, we just keep the BM25 scores
        if l == 1 or top_bm25_number == n:
            diversed_scores = QueryResult(bm25_scores.query, bm25_scores.documents[:n], bm25_scores.scores[:n])
        else:
            diversed_scores = self.diversify(bm25_scores, n=n, l=l)

        # 3: rerank the top n documents again with a neural ranker
        top_docs = diversed_scores.documents[:n]
        try:
            neural_scores = self.rerank_with_neural_model(query, top_docs)
        except Exception as e:
            print(f"An error occurred during reranking with the neural model: {e}")
            neural_scores = bm25_scores

        # 4 (Optional): combining scores to get the final ranking.
        final_ranking = neural_scores  # TODO: df: Implement. This is only optional if we want to ensemble rankings. Otherwise we can just delete it

        return final_ranking

    def _load_trained_model(self):
        model = TextEmbeddingModel(self.vocab_size, self.embedding_dim, self.max_length)
        model.load_state_dict(torch.load(get_path("ranker/model/rerank_model.pth")))
        model.eval()
        return model

    def rerank_with_neural_model(self, query: str, documents: list[DocumentEntry]):
        query_encoded = self.tokenizer.encode(query, add_special_tokens=False, max_length=self.max_length,
                                              padding='max_length', truncation=True)
        query_tensor = torch.tensor(query_encoded).unsqueeze(0)
        query_embedding = self.trained_ranker.forward(query_tensor)

        neural_scores = []
        for doc in documents:
            doc_encoded = self.tokenizer.encode(doc.page_text, add_special_tokens=False, max_length=self.max_length,
                                                padding='max_length', truncation=True)
            doc_tensor = torch.tensor(doc_encoded).unsqueeze(0)
            doc_embedding = self.trained_ranker(doc_tensor)

            # Reshape the embeddings to [max_length, embedding_dim] to calculate cosine similarity
            query_embedding_reshaped = query_embedding.view(-1,
                                                            self.embedding_dim)  # Shape: [max_length, embedding_dim]
            doc_embedding_reshaped = doc_embedding.view(-1, self.embedding_dim)  # Shape: [max_length, embedding_dim]

            cosine_sim = nn.functional.cosine_similarity(query_embedding_reshaped, doc_embedding_reshaped,
                                                         dim=-1).mean().item()
            neural_scores.append((doc, cosine_sim))

        neural_scores.sort(key=lambda x: x[1], reverse=True)

        ranked_documents = [doc for doc, score in neural_scores]
        scores = [score for doc, score in neural_scores]

        return QueryResult(query, ranked_documents, scores)

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


    def measure_diversity(self, unique_words: set, new_words: set, total_docs: int) -> float:
        combined_unique_words = unique_words.union(new_words)
        return len(combined_unique_words) / total_docs

    def diversify(self, ranking: QueryResult, n=100, l=1.0) -> QueryResult:
        class HeapElement:
            def __init__(self, importance, doc, score):
                self.importance = importance
                self.doc = doc
                self.score = score

            def __lt__(self, other):
                return self.importance > other.importance  # reverse for max-heap behavior

        reranked = []
        rescores = []
        unique_words = set()

        all_docs = ranking.documents
        all_scores = ranking.scores

        # Add the first entry
        most_relevant_doc = all_docs[0]
        most_relevant_score = all_scores[0]
        reranked.append(most_relevant_doc)
        rescores.append(most_relevant_score)
        unique_words.update(most_relevant_doc.enc_text)

        # Priority queue to store potential documents with their importance scores
        heap = []
        for doc, score in zip(all_docs[1:], all_scores[1:]):  # Skip the first doc as it's already added
            new_unique_words = set(doc.enc_text)
            importance = l * score + (1 - l) * self.measure_diversity(unique_words, new_unique_words, len(reranked) + 1)
            heapq.heappush(heap, HeapElement(-importance, doc, score))

        while len(reranked) < n and heap:
            most_important = heapq.heappop(heap)
            most_important_doc = most_important.doc
            most_important_score = most_important.score
            reranked.append(most_important_doc)
            rescores.append(most_important_score)
            unique_words.update(most_important_doc.enc_text)

            # Update the heap with new importances
            new_heap = []
            for doc, score in zip(all_docs, all_scores):
                if doc not in reranked:
                    new_unique_words = set(doc.enc_text)
                    importance = l * score + (1 - l) * self.measure_diversity(unique_words, new_unique_words,
                                                                              len(reranked) + 1)
                    heapq.heappush(new_heap, HeapElement(-importance, doc, score))
            heap = new_heap

        return QueryResult(ranking.query, reranked, rescores)

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


def setup_docker(sleep_time=15):
    os.system(f"""
        docker compose down;
        docker compose up -d --build db;
        sleep {sleep_time};
        """)


def main():
    # Setup Docker
    setup_docker()

    from ranker.Ranker import Ranker

    ranker = Ranker()
    all_docs = ranker.all_docs
    print(f"Total documents available: {len(all_docs)}")

    query = "food and drinks"  # Can be changed for experimenting
    index_length = len(
        all_docs)  # Use len(all_docs) to evaluate on full index OR a number, e.g., 1000, to only evaluate on the first 1000 documents
    n = 100

    # Prepare index.
    trun_docs = all_docs[:index_length]

    start_time = time.time()
    print(f"Ranking is starting...")

    query_result = ranker.rank_query(query, documents=trun_docs, n=n)

    end_time = time.time()
    print(f"Query: '{query}', Number of documents: {index_length}, Time required: {end_time - start_time} seconds")

    urls = [doc.url for doc in query_result.documents]
    data = {'query': query_result.query, 'urls': urls, 'scores': query_result.scores}

    df = pd.DataFrame(data)

    print(f"Size of dataset: {len(df)}. Is it the proper size? {n == len(df)}")

    pd.set_option('display.max_rows', 100)  # Ensure at least 100 rows are displayed
    display(df.head(n=n))


if __name__ == "__main__":
    main()
