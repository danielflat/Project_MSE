import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BatchEncoding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import csv
import numpy as np

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
    ranker = Ranker(model_path='./local_model/model', tokenizer_path='./local_model/tokenizer', device='cpu')

    # Fine-tune the model with the labeled data
    ranker.fine_tune_model(labeled_data_file)