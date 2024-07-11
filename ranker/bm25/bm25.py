import math
import numpy as np
from multiprocessing import Pool, cpu_count
import pandas as pd
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import re
import spacy
from transformers import BertTokenizer
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import math

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
nlp = spacy.load('en_core_web_sm')
lemmatizer = WordNetLemmatizer()


# Tokenizer functions for text processing
def tokenizer(doc):
    return doc.split()


def spacy_tokenizer(doc):
    return [token.text for token in nlp(doc)]


def nltk_lemmatizer(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]


def bert_subword_tokenizer(tokens):
    subwords = []
    for token in tokens:
        subwords.extend(bert_tokenizer.tokenize(token))
    return subwords


def combined_tokenizer(doc):
    '''
    Tokenizes the input document using a combination of spaCy, NLTK, and BERT tokenizers.
    :param doc: str: Input document
    :return: list: List of subword tokens
    '''
    spacy_tokens = spacy_tokenizer(doc)

    # Step 2: Use NLTK for lemmatization
    lemmatized_tokens = nltk_lemmatizer(spacy_tokens)

    # Step 3: Use BERT for subword tokenization
    bert_tokens = bert_subword_tokenizer(lemmatized_tokens)

    return bert_tokens



# Base Ranker class
class Ranker(ABC):
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


# BM25 and variants implementation
class BM25:
    def __init__(self, corpus, tokenizer=None):
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer

        if tokenizer:
            corpus = self._tokenize_corpus(corpus)

        nd = self._initialize(corpus)
        self._calc_idf(nd)

    def _initialize(self, corpus):
        nd = {}
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
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

        '''pool = Pool(cpu_count())
        tokenized_corpus = pool.map(self.tokenizer, corpus)
        pool.close()
        pool.join()'''

        print('Tokenization successful.')
        tokenized_corpus = [self.tokenizer(doc) for doc in corpus]
        return tokenized_corpus




    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()

    def get_batch_scores(self, query, doc_ids):
        raise NotImplementedError()

    def get_top_n(self, query, documents, n=5):
        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"
        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]


class BM25Okapi(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        idf_sum = 0
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        print(f"Query: {query}, Scores: {score}")
        return score

class BM25L(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=0.5):
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
        print(f"Query: {query}, Scores: {score}")
        return score

class BM25Plus(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=1):
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            idf = math.log(self.corpus_size + 1) - math.log(freq)
            self.idf[word] = idf

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
                                               (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
        print(f"Query: {query}, Scores: {score}")
        return score

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
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
        self.bias = nn.Parameter(torch.zeros(1))  # Add bias as a parameter

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        x += self.bias
        x = torch.relu(x)
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



# StatisticalRanker class to wrap the BM25 variants
class StatisticalRanker(Ranker):
    def __init__(self, corpus, ranker_type="bm25_okapi", tokenizer=None, **kwargs):
        self.corpus = corpus
        self.tokenizer = tokenizer or tokenizer
        self.ranker = self._initialize_ranker(ranker_type, **kwargs)
        self.ranker_name = ranker_type

    def _initialize_ranker(self, ranker_type, **kwargs):
        if ranker_type != "neural" and kwargs.get("idf_model"):
            raise ValueError("IDF model not required for statistical rankers. Please remove the 'idf_model' argument.")
        if ranker_type == "bm25_okapi":
            return BM25Okapi(self.corpus, self.tokenizer, **kwargs)
        elif ranker_type == "bm25_plus":
            return BM25Plus(self.corpus, self.tokenizer, **kwargs)
        elif ranker_type == "bm25_l":
            return BM25L(self.corpus, self.tokenizer, **kwargs)
        elif ranker_type == "neural":
            idf_model = kwargs.get("idf_model")
            return NeuralBM25(corpus=self.corpus, tokenizer=self.tokenizer, **kwargs)
        else:
            raise ValueError(f"Unknown ranker type: {ranker_type}. Valid options are 'bm25_okapi', 'bm25_plus', 'bm25_l', 'neural'.")

    def get_scores(self, query):
        return self.ranker.get_scores(query)

    def get_top_n(self, query, documents, n=5):
        return self.ranker.get_top_n(query, documents, n)


# RankerFactory to create rankers
class RankerFactory:
    @staticmethod
    def create_ranker(ranker_type, corpus=None, tokenizer=None, **kwargs):
        if ranker_type != "neural" and kwargs.get("idf_model"):
            raise ValueError("IDF model not required for statistical rankers. Please remove the 'idf_model' argument.")
        if ranker_type in ['bm25_okapi', 'bm25_plus', 'bm25_l', 'neural']:
            return StatisticalRanker(corpus, ranker_type=ranker_type, tokenizer=tokenizer, **kwargs)
        else:
            raise ValueError(f"Unknown ranker type: {ranker_type}")


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



if __name__ == '__main__':
    corpus_df = pd.read_csv('../../dummyindex.csv', delimiter=',')
    corpus = corpus_df['text'].tolist()
    # tokenized_corpus = [doc.split() for doc in corpus]
    tokenized_corpus = [tokenizer(doc) for doc in corpus]

    # test statistical ranker
    for ranker_type in ['bm25_okapi', 'bm25_plus', 'bm25_l']:
        print(ranker_type)
        # ranker_type = 'bm25_plus'  # Change to 'bm25_plus', 'bm25_l', 'tfidf', or 'neural' as needed
        ranker = RankerFactory.create_ranker(ranker_type, corpus=corpus, tokenizer=tokenizer)

        query = ['Statue', 'of','Liberty']

        # Retrieve the top N ranked documents
        top_n_documents = ranker.get_top_n(query, corpus, n=5)
        print(f"{ranker_type} Top Documents:")
        for doc in top_n_documents:
            print(doc)

    # test neural ranker
    idf_scores = calculate_idf(tokenized_corpus)
    training_data = prepare_training_data(tokenized_corpus, idf_scores)
    idf_model = train_idf_model(training_data)
    neural_idf_scores = generate_neural_idf_scores(corpus, idf_model, tokenizer)
    label_corpus_with_idf(corpus_df, neural_idf_scores)

    # ranker_type = 'neural'
    ranker = RankerFactory.create_ranker('neural', corpus=corpus, tokenizer=tokenizer, idf_model=idf_model)
    query = ['Statue', 'of','Liberty']
    top_n_documents = ranker.get_top_n(query, corpus, n=5)
    print("Neural Top Documents:")
    for doc in top_n_documents:
        print(doc)
