import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
#from ranker.TextEmbeddings import TextDataset, TextEmbeddingModel, CosineSimilarityLoss
from transformers import BertTokenizer
import pandas as pd


def load_data(fold_files):
    data_frames = [pd.read_csv(file) for file in fold_files]
    return pd.concat(data_frames, ignore_index=True)


# Vocabulary Building
def build_vocab(data):
    queries = data['query'].tolist()
    documents = data['page_text'].tolist()
    non_relevant_documents = data['page_text_non_rel_document'].tolist()

    tokenized_queries = [q.lower().split() for q in queries]
    tokenized_documents = [d.lower().split() for d in documents]
    tokenized_non_relevant_documents = [d.lower().split() for d in non_relevant_documents]

    word_list = list(set([word for sublist in tokenized_queries + tokenized_documents + tokenized_non_relevant_documents for word in sublist]))
    vocab = {word: i + 1 for i, word in enumerate(word_list)}

    return vocab, len(vocab) + 1


def prepare_dataloaders(train_files, test_files, vocab, max_length, batch_size):
    train_data = load_data(train_files)
    test_data = load_data(test_files)

    assert isinstance(train_data, pd.DataFrame), "train_data should be a pandas DataFrame"
    assert isinstance(test_data, pd.DataFrame), "test_data should be a pandas DataFrame"


    train_dataset = TextDataset(train_data, vocab, max_length)
    test_dataset = TextDataset(test_data, vocab, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    return train_loader, test_loader


def train_model(data, vocab, max_length, num_epochs=10, batch_size=32, learning_rate=0.001):
    model = TextEmbeddingModel()
    criterion = CosineSimilarityLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    dataset = TextDataset(data, vocab, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0
        for i, (query, doc, doc_non_rel, rel_label, non_rel_label) in enumerate(dataloader):
            optimizer.zero_grad()

            query_embeddings = model(query)
            doc_embeddings = model(doc)
            doc_non_rel_embeddings = model(doc_non_rel)

            rel_loss = criterion(query_embeddings, doc_embeddings, rel_label)
            non_rel_loss = criterion(query_embeddings, doc_non_rel_embeddings, non_rel_label)

            loss = rel_loss + non_rel_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}')

    return model


def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, inputs.float())  # Example loss function
            running_loss += loss.item()
    print(f"Test Loss: {running_loss / len(dataloader)}")

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = tokenizer.get_vocab()
    max_length = 512
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    train_files = [f"../data/traindata_1000_{i}_easy.csv" for i in range(8)]  # Use 8 folds for training
    test_files = ["../data/traindata_1000_8_easy.csv"]  # Use 1 fold for testing

    train_loader, test_loader = prepare_dataloaders(train_files, test_files, tokenizer, max_length, batch_size)

    model = train_model(train_loader, vocab, max_length, num_epochs, batch_size, learning_rate)
    evaluate_model(model, test_loader, nn.MSELoss())


if __name__ == '__main__':
    pass
    # main()


class TextDataset(Dataset):
    def __init__(self, data, vocab, max_length):
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
        query = self.data.iloc[idx]['query']
        document = self.data.iloc[idx]['page_text']
        document_non_relevant = self.data.iloc[idx]['page_text_non_rel_document']
        relevance_label = 1  # for the relevant document
        non_relevance_label = 0  # for the non-relevant document

        query_encoded = self.encode_text(query)
        document_encoded = self.encode_text(document)
        document_non_relevant_encoded = self.encode_text(document_non_relevant)

        return query_encoded, document_encoded, document_non_relevant_encoded, relevance_label, non_relevance_label


# Initialize BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
data = pd.read_csv('../data/traindata_rand_1000_0.csv')

# Tokenization and vocabulary building
queries = data['query'].tolist()
documents = data['page_text'].tolist()
non_relevant_documents = data['page_text_non_rel_document'].tolist()

# Flatten the lists to create a combined vocabulary
tokenized_queries = [q.lower().split() for q in queries]
tokenized_documents = [d.lower().split() for d in documents]
tokenized_non_relevant_documents = [d.lower().split() for d in non_relevant_documents]

word_list = list(set([word for sublist in tokenized_queries + tokenized_documents + tokenized_non_relevant_documents for word in sublist]))
vocab = {word: i+1 for i, word in enumerate(word_list)}  # starting from 1; 0 will be used for padding


# Create the dataset and dataloader
data = pd.read_csv('../data/traindata_rand_1000_0.csv')
vocab_size = 30522
embed_dim = 128
max_length = 512

dataset = TextDataset(data, vocab, max_length)
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

# Example usage
data = pd.read_csv('../data/traindata_rand_1000_0.csv')
# Tokenization and vocabulary building
queries = data['query'].tolist()
documents = data['page_text'].tolist()
non_relevant_documents = data['page_text_non_rel_document'].tolist()

# Flatten the lists to create a combined vocabulary
tokenized_queries = [q.lower().split() for q in queries]
tokenized_documents = [d.lower().split() for d in documents]
tokenized_non_relevant_documents = [d.lower().split() for d in non_relevant_documents]

word_list = list(set([word for sublist in tokenized_queries + tokenized_documents + tokenized_non_relevant_documents for word in sublist]))
vocab = {word: i+1 for i, word in enumerate(word_list)}  # starting from 1; 0 will be used for padding

# adjust dimensions to BERT model


# Helper function to encode text to padded indices
def encode_text(text, max_len):
    tokens = text.lower().split()
    indices = [vocab.get(token, 0) for token in tokens]  # get index from vocab, default to 0 (padding)
    indices = indices[:max_len]  # truncate if necessary
    indices += [0] * (max_len - len(indices))  # pad with zeros
    return torch.tensor(indices, dtype=torch.long)
