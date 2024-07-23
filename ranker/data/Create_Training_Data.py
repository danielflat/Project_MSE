#Create Training Data
#Importing BM25 module failed so I copied our basic BM25 class into this file
# Train Model for Reranking
# Train on ClueWeb and MS Marco
# Author: Lilli Diederichs
# Credits for the ColBertv2 introduction:
# https://jina.ai/news/what-is-colbert-and-late-interaction-and-why-they-matter-in-search/

# Format of train dataset
# id, url, page_text, query, rel_label, page_text_non_rel_document, nonrel_label

import keybert
import pandas as pd
from keybert import KeyBERT
import numpy as np
import math
import os
from collections import Counter

# Initialize KeyBERT model
kw_model = KeyBERT()

def load_data(filepath):
    """Load dataset from an Excel file."""
    return pd.read_excel(filepath)

def create_subsets(data, subset_size):
    """Split data into subsets of specified size."""
    return np.array_split(data, math.ceil(len(data) / subset_size))

def extract_keywords(text, model):
    """Extract keywords from text using KeyBERT."""
    return model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words=None)

def create_query_dataframe(dataset, model):
    """Create a DataFrame with queries and related documents."""
    columns = ['id', 'url', 'page_text', 'query', 'rel_label', 'page_text_non_rel_document', 'nonrel_label']
    result_df = pd.DataFrame(columns=columns)
    
    for _, row in dataset.iterrows():
        doc_id = row['id']
        url = row['url']
        page_text = row['page_text']
        summaries = extract_keywords(page_text, model)
        
        for query, _ in summaries:
            new_row = {
                'id': doc_id,
                'url': url,
                'page_text': page_text,
                'query': query,
                'rel_label': 1,
                'page_text_non_rel_document': ' ',  # placeholder
                'nonrel_label': 0
            }
            result_df = result_df.append(new_row, ignore_index=True)
    
    return result_df

def add_non_relevant_docs(result_df, corpus, keywords):
    """Add non-relevant documents to the DataFrame."""
    for index, row in result_df.iterrows():
        query = row['query']
        non_rel_docs = [doc for doc in corpus if doc != row['page_text'] and any(keyword in doc for keyword in keywords)]
        if non_rel_docs:
            non_rel_doc = np.random.choice(non_rel_docs)
            result_df.at[index, 'page_text_non_rel_document'] = non_rel_doc
    return result_df

class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.avg_doc_length = sum(len(doc) for doc in documents) / len(documents)
        self.doc_freqs = self._compute_doc_freqs()
        self.idf = self._compute_idf()

    def _compute_doc_freqs(self):
        """Compute document frequencies for each term."""
        doc_freqs = Counter()
        for doc in self.documents:
            unique_terms = set(doc)
            for term in unique_terms:
                doc_freqs[term] += 1
        return doc_freqs

    def _compute_idf(self):
        """Compute inverse document frequency for each term."""
        idf = {}
        total_docs = len(self.documents)
        for term, freq in self.doc_freqs.items():
            idf[term] = math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
        return idf

    def score(self, query, document):
        """Compute BM25 score for a single document given a query."""
        doc_len = len(document)
        doc_counter = Counter(document)
        score = 0
        for term in query:
            if term in doc_counter:
                term_freq = doc_counter[term]
                idf = self.idf.get(term, 0)
                score += idf * (term_freq * (self.k1 + 1)) / (term_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length))
        return score

    def score_documents(self, query):
        """Compute BM25 score for each document in the corpus given a query."""
        return [self.score(query, doc) for doc in self.documents]

def add_bm25_non_relevant_docs(result_df, corpus):
    """Add non-relevant documents using BM25 scoring."""
    bm25 = BM25(corpus)
    
    for index, row in result_df.iterrows():
        query = row['query'].split()
        scores = bm25.score_documents(query)
        min_score_index = scores.index(min(scores))
        non_rel_doc = ' '.join(corpus[min_score_index])
        result_df.at[index, 'page_text_non_rel_document'] = non_rel_doc
    
    return result_df

def main():
    filepath = 'crawled_data_backup_2024-07-08_09-57-39_CONCAT.xlsx'
    data = load_data(filepath)
    
    subsets = create_subsets(data, 200)
    print(f'Total subsets: {len(subsets)}')
    
    tübingen_keywords = ['Tübingen', 'Tuebingen', 'Baden-Württemberg', 'Baden Wuerttemberg']
    
    for i, subset in enumerate(subsets):
        print(f'Processing subset {i+1}/{len(subsets)}')
        
        result_df = create_query_dataframe(subset, kw_model)
        corpus = subset['page_text'].tolist()
        
        result_df = add_non_relevant_docs(result_df, corpus, tübingen_keywords)
        result_df.to_csv(f'traindata_rand_1000_{i}.csv', index=False)
        
        corpus_tokens = [doc.split() for doc in corpus]
        result_df = add_bm25_non_relevant_docs(result_df, corpus_tokens)
        result_df.to_csv(f'traindata_1000_easy_{i}.csv', index=False)
        
        print(f'Done with subset {i+1}')

if __name__ == '__main__':
    main()
