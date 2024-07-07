import csv
import os
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments, BertTokenizer

from db.DocumentRepository import DocumentRepository


class Ranker:

    def __init__(self):
        self.documentRepository = DocumentRepository()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def rank_query(self, query):
        tokenized_query = self.tokenizer.encode(query)
        documents_vectors = self.documentRepository.getEncodedTextOfAllDocuments()
        bm25_scores = self.rank_BM25(tokenized_query, documents_vectors)
        return bm25_scores

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

        # TODO: Can be precomputed ------
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

        sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        return sorted_scores

def tokenizer(doc):
    return doc.split()

class BM25:
    '''
    Ranker class for BM25 ranking model

    '''
    def __init__(self, corpus, tokenizer=None):
        self.corpus_size = 0
        self.tokenizer = tokenizer
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []

        if tokenizer:
            corpus = self._tokenize_corpus(corpus)

        nd = self._initialize(corpus)
        self._calc_idf(nd)

    def _initialize(self, corpus):
        nd = {}
        num_doc = 0

        for doc in corpus:
            self.doc_len.append(len(doc))
            num_doc += len(doc)
            frequencies = {}
            for word in doc:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                try:
                    nd[word] += 1
                except KeyError:
                    nd[word] = 1

            self.corpus_size += 1

        self.avgdl = num_doc / self.corpus_size
        return nd

    def _tokenize_corpus(self, corpus):
        pool = Pool(cpu_count())
        tokenized_corpus = pool.map(self.tokenizer, corpus)
        return tokenized_corpus

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()

    def get_batch_scores(self, query, doc_ids):
        raise NotImplementedError()

    def get_top_n(self, query, documents, n=5):
        assert self.corpus_size == len(documents), "The number of documents should be equal to the corpus size"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]


class BM25Plus(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=1):
        # Algorithm specific parameters
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            idf = np.log(self.corpus_size + 1) - np.log(freq)
            self.idf[word] = idf

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
                                               (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
                                               (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
        return score.tolist()


class BM25Okapi(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        """
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
        """
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in nd.items():
            idf = np.log(self.corpus_size - freq + 0.5) - np.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        """
        The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
        :param query:
        :return:
        """
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score.tolist()


class BM25L(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=0.5):
        # Algorithm specific parameters
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            idf = np.log(self.corpus_size + 1) - np.log(freq + 0.5)
            self.idf[word] = idf

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
            score += (self.idf.get(q) or 0) * (self.k1 + 1) * (ctd + self.delta) / \
                     (self.k1 + ctd + self.delta)
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
            score += (self.idf.get(q) or 0) * (self.k1 + 1) * (ctd + self.delta) / \
                     (self.k1 + ctd + self.delta)
        return score.tolist()


def search(query, corpus, ranker):
    # Retrieve the top N ranked documents
    search1 = set(ranker.get_top_n(query, corpus, n=5))

    search2 = set(ranker.get_top_n(query[0].split(" "), corpus, n=5))

    return search2.intersection(search1)









class NeuralRanker:
    def __init__(self, model_path: str, tokenizer_path: str, device: str = 'cpu'):
        self.device = device
        print(f"Loading tokenizer from {tokenizer_path}")
        print(f"Loading model from {model_path}")

        # Check if paths exist
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer path not found: {tokenizer_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def load_labeled_data(self, file_path):
        queries, passages, labels = [], [], []
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                queries.append(row[0])
                passages.append(row[1])
                labels.append(int(row[2]))
        return queries, passages, labels

    def fine_tune_model(self, labeled_data_file):
        # Read the labeled data from file
        queries, passages, labels = self.load_labeled_data(labeled_data_file)

        # Create a custom dataset
        class CustomDataset(Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        encodings = self.tokenizer(queries, passages, truncation=True, padding=True, max_length=192)
        dataset = CustomDataset(encodings, labels)

        # Define training arguments for CPU
        training_args = TrainingArguments(
            output_dir='./results',          # output directory
            num_train_epochs=3,              # total number of training epochs
            per_device_train_batch_size=8,   # batch size per device during training
            per_device_eval_batch_size=16,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=10,
            no_cuda=True,                    # Force training on CPU
        )

        # Define a Trainer
        trainer = Trainer(
            model=self.model,                 # the instantiated Transformers model to be trained
            args=training_args,               # training arguments, defined above
            train_dataset=dataset,            # training dataset
            compute_metrics=self.compute_metrics
        )

        # Train the model
        trainer.train()

        # Save the fine-tuned model
        self.model.save_pretrained('./fine_tuned_model')
        self.tokenizer.save_pretrained('./fine_tuned_model')

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }





if __name__ == "__main__":
    # Path to your labeled CSV file
    labeled_data_file = 'labeled_data.csv'

    # Initialize Ranker with local paths
    #ranker = NeuralRanker(model_path='./local_model/model', tokenizer_path='./local_model/tokenizer', device='cpu')

    # Fine-tune the model with the labeled data
    #ranker.fine_tune_model(labeled_data_file)

    # Read the CSV file
    corpus_df = pd.read_csv('../dummyindex.csv', delimiter=',')

    # Convert the corpus into a list of strings (documents)
    corpus = corpus_df['text'].tolist()  # Replace 'text_column_name' with the actual column name in your CSV

    # Tokenize the corpus
    tokenized_corpus = [tokenizer(doc) for doc in corpus]

    # Initialize the BM25Plus ranker
    ranker = BM25Okapi(tokenized_corpus)

    # Define your query
    query = ['Statue', 'of', 'Liberty']
    # query = ['Statue of Liberty']

    # Retrieve the top N ranked documents
    top_n_documents = ranker.get_top_n(query, corpus, n=5)

    for doc in top_n_documents:
        print(doc)

    # Search for documents that are common between two different search methods
    query = ['Statue of Liberty']
    common_docs = search(query, corpus, ranker)
    for doc in common_docs:
        print(doc)


