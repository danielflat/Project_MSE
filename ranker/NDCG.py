import numpy as np


class NDCG():

    def __init__(self):
        pass

    def dcg(self, relevance_scores, n=100):
        """
        Calculate DCG (Discounted Cumulative Gain) at rank n (only consider the top n documents).
        This involves summing the relevance scores of the results, discounted logarithmically by their position in the list.

        Returns DCG score for the top n documents.
        """

        relevance_scores = np.asarray(relevance_scores)[:n]
        if relevance_scores.size:
            return np.sum(relevance_scores / np.log2(np.arange(2, relevance_scores.size + 2)))
        return 0.0

    def idcg(self, relevance_scores, n=100):
        """
        Calculate IDCG (Ideal Discounted Cumulative Gain) at rank n (only consider the top n documents).
        This is the DCG of the best possible ranking. In other words, the ground truth ranking.

        Returns IDCG score for the top n documents.
        """

        sorted_scores = sorted(relevance_scores, reverse=True)
        return self.dcg(sorted_scores, n)

    def ndcg(self, relevance_scores: list[float], n=100):
        """
        Calculated the Normalized Discounted Cumulative Gain (nDCG) at rank n.

        Returns NDCG score for the top n documents.
        """

        dcg_max = self.idcg(relevance_scores, n)
        if not dcg_max:
            return 0.0
        return self.dcg(relevance_scores, n) / dcg_max
