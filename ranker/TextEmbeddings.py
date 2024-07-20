import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, data, vocab, max_length):
        assert isinstance(data, pd.DataFrame), "data should be a pandas DataFrame"
        self.data = data
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def encode_text(self, text):
        tokens = text.lower().split()
        indices = [self.vocab.get(token, 0) for token in tokens]  # get index from vocab, default to 0 (padding)
        indices = indices[:self.max_length]  # truncate if necessary
        indices += [0] * (self.max_length - len(indices))  # pad with zeros
        return torch.tensor(indices, dtype=torch.long)

    def __getitem__(self, idx):
        print(self.data.__class__)
        query = self.data.iloc[idx]['query']
        document = self.data.iloc[idx]['page_text']
        document_non_relevant = self.data.iloc[idx]['page_text_non_rel_document']
        relevance_label = 1  # for the relevant document
        non_relevance_label = 0  # for the non-relevant document

        query_encoded = self.encode_text(query)
        document_encoded = self.encode_text(document)
        document_non_relevant_encoded = self.encode_text(document_non_relevant)

        return query_encoded, document_encoded, document_non_relevant_encoded, relevance_label, non_relevance_label

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)


    def mean_without_zeros(self, tensor, dim):
        sum_tensor = tensor.sum(dim=dim)
        num_non_zeros = (tensor != 0).sum(dim=dim)
        return sum_tensor / num_non_zeros

    def forward(self, query_embeddings, doc_embeddings, labels):
        cosine_sim = self.cosine_similarity(self.mean_without_zeros(query_embeddings, dim=1),
                                            self.mean_without_zeros(doc_embeddings, dim=1))  # check dims, (B, V)
        loss = nn.MSELoss()(cosine_sim, labels)
        return loss


class TextEmbeddingModel(nn.Module):
    def __init__(self, vocab_size=30522, embedding_dim=128, max_length=512):
        super(TextEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.max_length = max_length

    def forward(self, x):
        x = self.embedding(x)
        # maybe delete the following line
        x = x.mean(dim=1)
        return x


class TextEmbeddingModelwithLSTM(nn.Module):
    def __init__(self, hidden_dim, num_layers, vocab_size=30522, embedding_dim=128, max_length=512):
        super(TextEmbeddingModelwithLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.max_length = max_length

    def forward(self, query, document, document_non_relevant):
        query_embedding = self.embedding(query)
        document_embedding = self.embedding(document)
        document_non_relevant_embedding = self.embedding(document_non_relevant)

        query_embedding = query_embedding.view(-1, self.max_length, -1)
        document_embedding = document_embedding.view(-1, self.max_length, -1)
        document_non_relevant_embedding = document_non_relevant_embedding.view(-1, self.max_length, -1)

        query_lstm, _ = self.lstm(query_embedding)
        document_lstm, _ = self.lstm(document_embedding)
        document_non_relevant_lstm, _ = self.lstm(document_non_relevant_embedding)

        return query_lstm, document_lstm, document_non_relevant_lstm


def mean_without_zeros(tensor, dim):
    sum_tensor = tensor.sum(dim=dim)
    num_non_zeros = (tensor != 0).sum(dim=dim)
    return sum_tensor / num_non_zeros