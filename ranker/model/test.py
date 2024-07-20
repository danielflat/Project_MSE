import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

# Load Data
data = pd.read_csv('../data/traindata_rand_1000_0.csv')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Vocabulary Building
def build_vocab(data):
    queries = data['query'].tolist()
    documents = data['page_text'].tolist()
    non_relevant_documents = data['page_text_non_rel_document'].tolist()

    tokenized_queries = [q.lower().split() for q in queries]
    tokenized_documents = [d.lower().split() for d in documents]
    tokenized_non_relevant_documents = [d.lower().split() for d in non_relevant_documents]

    word_list = list(
        set([word for sublist in tokenized_queries + tokenized_documents + tokenized_non_relevant_documents for word in
             sublist]))
    vocab = {word: i + 1 for i, word in enumerate(word_list)}

    return vocab, len(vocab) + 1


vocab, vocab_size = build_vocab(data)
print(f'Vocabulary size: {vocab_size}')


# Model Definitions
class TextEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=50):
        super(TextEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)

    def forward(self, x):
        return self.embedding(x)


class ConvTextModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(ConvTextModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.mean(x, 2)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def mean_without_zeros(self, tensor, dim):
        sum_tensor = tensor.sum(dim=dim)
        num_non_zeros = (tensor != 0).count_nonzero(dim=dim)
        return sum_tensor / num_non_zeros

    def forward(self, query_embeddings, doc_embeddings, labels):
        cosine_sim = self.cosine_similarity(self.mean_without_zeros(query_embeddings, dim=1),
                                            self.mean_without_zeros(doc_embeddings, dim=1))
        loss = nn.MSELoss()(cosine_sim, labels)
        return loss


# Dataset and DataLoader
class TextDataset(Dataset):
    def __init__(self, data, vocab, max_length):
        self.data = data
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def encode_text(self, text):
        tokens = text.lower().split()
        indices = [self.vocab.get(token, 0) for token in tokens]
        indices = indices[:self.max_length]
        indices += [0] * (self.max_length - len(indices))
        return torch.tensor(indices, dtype=torch.long)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        query = self.encode_text(row['query'])
        document = self.encode_text(row['page_text'])
        document_non_relevant = self.encode_text(row['page_text_non_rel_document'])
        return query, document, document_non_relevant


max_length = 512
dataset = TextDataset(data, vocab, max_length)
dataloader = DataLoader(dataset, batch_size=50, shuffle=True)


# Training the Model
def train_model(model, dataloader, num_epochs, loss_function, optimizer):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for queries, docs, non_relevant_docs in dataloader:
            query_embeddings = model(queries)
            doc_embeddings = model(docs)
            non_relevant_doc_embeddings = model(non_relevant_docs)

            relevant_labels = torch.ones(query_embeddings.size(0)).to(query_embeddings.device)
            non_relevant_labels = torch.zeros(query_embeddings.size(0)).to(query_embeddings.device)

            loss_relevant = loss_function(query_embeddings, doc_embeddings, relevant_labels)
            loss_non_relevant = loss_function(query_embeddings, non_relevant_doc_embeddings, non_relevant_labels)

            loss = loss_relevant + loss_non_relevant
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        average_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')


embed_dim = 128
vocab_size = bert_tokenizer.vocab_size
embedding_model = TextEmbeddingModel(vocab_size, embed_dim)
optimizer = optim.Adam(embedding_model.parameters(), lr=0.001)
loss_function = CosineSimilarityLoss()
num_epochs = 10

train_model(embedding_model, dataloader, num_epochs, loss_function, optimizer)

# Save the Model

torch.save(embedding_model.state_dict(), 'rerank_model.pth')


# Re-rank the Dataset
def rerank_data(model, data, dataset):
    model.eval()
    reranked_results = []
    with torch.no_grad():
        for idx, row in data.iterrows():
            query_encoded = dataset.encode_text(row['query']).unsqueeze(0)
            doc_encoded = dataset.encode_text(row['page_text']).unsqueeze(0)
            document_non_relevant_encoded = dataset.encode_text(row['page_text_non_rel_document']).unsqueeze(0)

            query_embedding = model(query_encoded)
            doc_embedding = model(doc_encoded)
            document_non_relevant_embedding = model(document_non_relevant_encoded)

            cosine_sim = nn.functional.cosine_similarity(query_embedding, doc_embedding, dim=-1).mean().item()
            reranked_results.append((idx, cosine_sim))

    reranked_results.sort(key=lambda x: x[1], reverse=True)
    reranked_data = data.iloc[[x[0] for x in reranked_results]].copy()
    reranked_data['cosine_similarity'] = [x[1] for x in reranked_results]

    reranked_data.to_csv('reranked_traindata.csv', index=False)
    print("Reranked results saved to reranked_traindata.csv")
    return reranked_data


reranked_data = rerank_data(embedding_model, data, dataset)
print(reranked_data)
