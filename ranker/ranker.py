import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BatchEncoding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
from collections import defaultdict
import csv
import re
from multiprocessing import Pool, cpu_count
import numpy as np
from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
import math


DEBUG = True


def tokenizer(doc):
    return doc.split()


class Ranker(ABC):
    """
    Abstract base class for all Rankers.
    """

    @abstractmethod
    def get_scores(self, query):
        pass

    @abstractmethod
    def get_top_n(self, query, documents, n=5):
        pass

    def search(self, query, corpus, n=5):
        search1 = set(self.get_top_n(query, corpus, n))
        search2 = set(self.get_top_n(query[0].split(" "), corpus, n))
        return search2.intersection(search1)


class BM25(Ranker):
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
                if word in nd:
                    nd[word] += 1
                else:
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
            if q not in self.idf:
                continue
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf[q] or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
                                           (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
        if DEBUG:
            print(f"Query: {query}, Scores: {score}")
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

        if DEBUG:
            print(f"Query: {query}, Scores: {score}")

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

        if DEBUG:
            print(f"Query: {query}, Scores: {score}")

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


class TfidfRanker(Ranker):
    def __init__(self, corpus):
        self.vectorizer = TfidfVectorizer(tokenizer=tokenizer, token_pattern=None)
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def get_scores(self, query):
        query_vec = self.vectorizer.transform([' '.join(query)])
        scores = np.dot(query_vec, self.tfidf_matrix.T).toarray()[0]
        print(f"Query: {query}, Scores: {scores}")
        return scores

    def get_top_n(self, query, documents, n=5):
        scores = self.get_scores(query)
        top_n_indices = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n_indices]


class IDFDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, idf = self.data[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(idf, dtype=torch.float32)

class IDFNet(nn.Module):
    def __init__(self, input_size):
        super(IDFNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NeuralBM25(BM25):
    def __init__(self, corpus, tokenizer=None, idf_model=None, k1=1.5, b=0.75, delta=0.5):
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.idf_model = idf_model
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        # No need to calculate IDF traditionally
        pass

    def get_idf(self, word_freq):
        with torch.no_grad():
            features = torch.tensor([word_freq], dtype=torch.float32)
            idf = self.idf_model(features).item()
        return idf

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
            idf = self.get_idf(q_freq.mean())  # Using mean frequency for simplicity
            score += idf * (self.k1 + 1) * (ctd + self.delta) / (self.k1 + ctd + self.delta)

        print(f"Query: {query}, Scores: {score}")

        return score

    def get_batch_scores(self, query, doc_ids):
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
            idf = self.get_idf(q_freq.mean())  # Using mean frequency for simplicity
            score += idf * (self.k1 + 1) * (ctd + self.delta) / (self.k1 + ctd + self.delta)
        return score.tolist()

# Training data preparation (example)
def prepare_training_data(corpus, idf_values):
    training_data = []
    for word, idf in idf_values.items():
        freq = sum([1 for doc in corpus if word in doc])
        features = [freq]  # Add more features if necessary
        training_data.append((features, idf))
    return training_data

# Example training process
def train_idf_model(training_data, epochs=100, lr=0.001):
    dataset = IDFDataset(training_data)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = IDFNet(input_size=len(training_data[0][0]))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for features, idf in dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.squeeze(), idf)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

    return model


def calculate_idf(tokenized_corpus):
    num_docs = len(tokenized_corpus)
    doc_freqs = defaultdict(int)

    for doc in tokenized_corpus:
        unique_words = set(doc)
        for word in unique_words:
            doc_freqs[word] += 1

    idf_scores = {word: math.log(num_docs / (freq + 1)) + 1 for word, freq in doc_freqs.items()}
    return idf_scores


def generate_neural_idf_scores(corpus, idf_model, tokenizer):
    tokenized_corpus = [tokenizer(doc) for doc in corpus]
    word_freqs = defaultdict(int)
    for doc in tokenized_corpus:
        for word in doc:
            word_freqs[word] += 1

    neural_idf_scores = {}
    for word, freq in word_freqs.items():
        features = torch.tensor([[freq]], dtype=torch.float32)
        with torch.no_grad():
            idf = idf_model(features).item()
        neural_idf_scores[word] = idf

    return neural_idf_scores


def label_corpus_with_idf(corpus_df, neural_idf_scores):
    tokenized_corpus = [tokenizer(doc) for doc in corpus_df['text']]
    idf_labels = []
    for doc in tokenized_corpus:
        idf_score = sum(neural_idf_scores[word] for word in doc) / len(doc)  # Average IDF score
        idf_labels.append(idf_score)
    corpus_df['idf_score'] = idf_labels


class NeuralRanker(Ranker):
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


class StatisticalRanker(Ranker):
    def __init__(self, corpus, ranker_type="bm25_okapi", tokenizer=None, **kwargs):
        self.corpus = corpus
        self.tokenizer = tokenizer or tokenizer
        self.ranker = self._initialize_ranker(ranker_type, **kwargs)

    def _initialize_ranker(self, ranker_type, **kwargs):
        if ranker_type == "bm25_okapi":
            return BM25Okapi(self.corpus, self.tokenizer, **kwargs)
        elif ranker_type == "bm25_plus":
            return BM25Plus(self.corpus, self.tokenizer, **kwargs)
        elif ranker_type == "bm25_l":
            return BM25L(self.corpus, self.tokenizer, **kwargs)
        elif ranker_type == "tfidf":
            return TfidfRanker(self.corpus)
        else:
            raise ValueError(f"Unknown ranker type: {ranker_type}")

    def get_scores(self, query):
        return self.ranker.get_scores(query)

    def get_top_n(self, query, documents, n=5):
        return self.ranker.get_top_n(query, documents, n)


class RankerFactory:
    @staticmethod
    def create_ranker(ranker_type, corpus=None, model_path=None, tokenizer_path=None, device='cpu', **kwargs):
        if ranker_type in ['bm25_okapi', 'bm25_plus', 'bm25_l', 'tfidf']:
            return StatisticalRanker(corpus, ranker_type=ranker_type, **kwargs)
        elif ranker_type == 'neural':
            if not model_path or not tokenizer_path:
                raise ValueError("For neural ranker, 'model_path' and 'tokenizer_path' must be provided")
            return NeuralRanker(model_path, tokenizer_path, device)
        else:
            raise ValueError(f"Unknown ranker type: {ranker_type}")




if __name__ == "__main__":
    # Path to your labeled CSV file

    # Read the CSV file
    corpus_df = pd.read_csv('../dummyindex.csv', delimiter=',')
    corpus = corpus_df['text'].tolist()  # Replace 'text' with the actual column name in your CSV

    # Initialize the Ranker Factory
    for ranker_type in ['bm25_okapi', 'bm25_plus', 'bm25_l', 'tfidf']:
        print(ranker_type)
        # ranker_type = 'bm25_plus'  # Change to 'bm25_plus', 'bm25_l', 'tfidf', or 'neural' as needed
        ranker = RankerFactory.create_ranker(ranker_type, corpus=corpus, tokenizer=tokenizer)

        query = ['Statue','of', 'Liberty']

        # Retrieve the top N ranked documents
        top_n_documents = ranker.get_top_n(query, corpus, n=5)
        print(f"{ranker_type} Top Documents:")
        for doc in top_n_documents:
            print(doc)

    '''# Search for documents that are common between two different search methods
    query = ['Statue of Liberty']
    common_docs = ranker.search(query, corpus, n=5)
    print("Common Documents in Search:")
    for doc in common_docs:
        print(doc)'''


