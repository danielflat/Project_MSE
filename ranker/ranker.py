# author: @lenardrommel

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BatchEncoding, PreTrainedTokenizerFast
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
from keybert import KeyBERT
from sklearn.metrics.pairwise import cosine_similarity
import csv
import numpy as np

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

        if NORMALIZES_SCORES:
            scores = self.normalize_scores(np.array(scores))

        ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked_docs], scores

    @staticmethod
    def normalize_scores(scores):
        min_score = scores.min()
        max_score = scores.max()
        return (scores - min_score) / (max_score - min_score)

    @staticmethod
    def label_scores(scores):
        return np.array([1 if score > 0.5 else 0 for score in scores])

    def read_documents(self, file_path):
        documents = []
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                documents.append(row[0])
        return documents

    def summarize_and_label_documents(self, documents, min_keywords=1, max_keywords=2):
        assert min_keywords >= 1, "Minimum number of keywords must be greater or equal to 1"
        assert max_keywords >= min_keywords, "Maximum number of keywords must be greater or equal to minimum number of keywords"

        labeled_data = []
        for doc in documents:
            keywords = self.keybert_model.extract_keywords(doc, keyphrase_ngram_range=(1, max_keywords), stop_words=None)
            # Ensure the keywords are unique and sorted by importance (score)
            keywords = sorted(set(keywords), key=lambda x: x[1], reverse=True)

            # Use the most important keywords individually
            for word, _ in keywords[:2]:
                _, scores = self.rank_documents(word, documents)
                scores = self.normalize_scores(np.array(scores))
                labels = self.label_scores(scores)
                labeled_data.extend(zip([word] * len(documents), documents, labels))

            # Append additional keywords to the top two keywords
            for base_word, _ in keywords[:2]:
                for word, _ in keywords:
                    if word != base_word:
                        combined_query = f"{base_word} {word}"
                        _, scores = self.rank_documents(combined_query, documents)
                        scores = self.normalize_scores(np.array(scores))
                        labels = self.label_scores(scores)
                        labeled_data.extend(zip([combined_query] * len(documents), documents, labels))

        return labeled_data

    def save_labeled_data(self, labeled_data, file_path):
        # Read existing data from file if it exists
        existing_data = set()
        if os.path.exists(file_path):
            with open(file_path, mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    existing_data.add(tuple(row))

        # Combine new data with existing data, ensuring no duplicates
        combined_data = existing_data.union(set(map(tuple, labeled_data)))

        # Write the combined data back to the file
        with open(file_path, mode='w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            for data in combined_data:
                writer.writerow(data)

    def load_labeled_data(self, file_path):
        documents = []
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                documents.append(row[0])
        return documents

    def fine_tune_model(self, labeled_data_file):
        # Read the labeled data from file
        queries, passages, labels = [], [], []
        with open(labeled_data_file, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                queries.append(row[0])
                passages.append(row[1])
                labels.append(int(row[2]))

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

        # Define training arguments
        training_args = TrainingArguments(
            output_dir='./results',          # output directory
            num_train_epochs=3,              # total number of training epochs
            per_device_train_batch_size=8,   # batch size per device during training
            per_device_eval_batch_size=16,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=10,
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
    # Path to your CSV file
    file_path = '../dummyindex.csv'
    labeled_data_file = 'labeled_data.csv'

    # Initialize Ranker with local paths
    ranker = Ranker(model_path='./local_model/model', tokenizer_path='./local_model/tokenizer', device='cpu')

    # Read the documents
    documents = ranker.read_documents(labeled_data_file)

    # Fine-tune the model with the labeled data
    ranker.fine_tune_model(labeled_data_file)


    # Summarize and label the documents
    # labeled_data = ranker.summarize_and_label_documents(documents, min_keywords=1, max_keywords=2)

    # Save the labeled data to a file non-redundantly
    # ranker.save_labeled_data(labeled_data, labeled_data_file)

    # Print the labeled data
    '''for summary, doc, label in labeled_data:
        print(f"Query: {summary}\nDocument: {doc}\nLabel: {label}\n")
    '''