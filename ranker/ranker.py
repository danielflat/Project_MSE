#author: @lenardrommel

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BatchEncoding, PreTrainedTokenizerFast
from transformers.modeling_outputs import SequenceClassifierOutput

class Ranker:
    def __init__(self, model_path: str, tokenizer_path: str, device: str = 'cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

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
            scores = [(score - min(scores)) / (max(scores) - min(scores)) for score in scores]


        ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked_docs], scores