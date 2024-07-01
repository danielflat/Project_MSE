#author: @lenardrommel

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BatchEncoding, PreTrainedTokenizerFast
from transformers.modeling_outputs import SequenceClassifierOutput
import os
from keybert import KeyBERT
from sklearn.metrics.pairwise import cosine_similarity
import csv


NORMALIZES_SCORES = True

class Ranker:
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
        self.keybert_model = KeyBERT('distilbert-base-nli-mean-tokens')

    def encode(self, query: str, passage: str, title: str = '-') -> BatchEncoding:
        return self.tokenizer(query,
                              text_pair='{}: {}'.format(title, passage),
                              max_length=192,
                              padding=True,
                              truncation=True,
                              return_tensors='pt').to(self.device)

    def rank_documents(self, query: str, documents: list):
        scores = []
        with torch.no_grad():
            for doc in documents:
                batch_dict = self.encode(query, doc)
                outputs: SequenceClassifierOutput = self.model(**batch_dict, return_dict=True)
                score = outputs.logits[0].item()
                scores.append(score)

            # normalize scores
            if NORMALIZES_SCORES:
                scores = [(score - min(scores)) / (max(scores) - min(scores)) for score in scores]

            # make scores a probability distribution
            # scores = torch.nn.functional.softmax(torch.tensor(scores), dim=0).tolist()
            scores = [(score - min(scores)) / (max(scores) - min(scores)) for score in scores]

        ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked_docs], scores

    def read_documents(self, file_path):
        documents = []
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                documents.append(row[0])
        return documents

    def summarize_documents(self, documents, n_keywords=3):
        assert n_keywords >= 1, "Number of keywords must be greater equal to 1"
        summaries = []
        for doc in documents:
            keywords = self.keybert_model.extract_keywords(doc, keyphrase_ngram_range=(n_keywords - 1, n_keywords), stop_words=None)
            summaries.append(" ".join([word for word, _ in keywords]))

        return summaries

    def label_documents(self, summaries, documents):
        labeled_data = []
        for summary in summaries:
            ranked_docs, scores = self.rank_documents(summary, documents)

            labels = [1 if score > 0.5 else 0 for score in scores]  # Assume score > 0.5 as relevant
            labeled_data.extend(zip([summary] * len(documents), documents, labels))
        return labeled_data

    def fine_tune_model(self, labeled_data):
        # Fine-tuning logic here, depending on your specific needs and available data
        pass


if __name__ == "__main__":
    # Path to your CSV file
    file_path = 'documents.csv'

    # Initialize Ranker with local paths
    ranker = Ranker(model_path='./local_model/model', tokenizer_path='./local_model/tokenizer', device='cpu')

    # Read the documents
    documents = ranker.read_documents('../dummyindex.csv')

    labeled_data_list = []
    for n_keywords in range(1, 4):
        # Summarize the documents
        summaries = ranker.summarize_documents(documents, n_keywords)

        # Label the documents based on summaries
        labeled_data = ranker.label_documents(summaries, documents)

        labeled_data_list.append(labeled_data)

    # Fine-tune the model with labeled data
    ranker.fine_tune_model(labeled_data_list)

    # Print the labeled data
    for summary, doc, label in labeled_data_list:
        print(f"Query: {summary}\nDocument: {doc}\nLabel: {label}\n")
