import os

import pandas as pd
from flask import Flask, render_template
from flask_bootstrap import Bootstrap

from ranker.ranker import BM25Okapi

app = Flask(__name__)
Bootstrap(app)

# Get the absolute path to the current directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute paths to the model and tokenizer directories
model_path = os.path.join(base_dir, 'ranker/local_model/model')
tokenizer_path = os.path.join(base_dir, 'ranker/local_model/tokenizer')

# Print paths to ensure correctness
print(f"Current working directory: {base_dir}")
print(f"Model path: {model_path}")
print(f"Tokenizer path: {tokenizer_path}")

# Check if paths exist
print("Checking if paths exist...")
print(f"Model path exists: {os.path.exists(model_path)}")
print(f"Tokenizer path exists: {os.path.exists(tokenizer_path)}")

# Example documents - in practice, fetch documents based on the search query
documents = pd.read_csv('dummyindex.csv', delimiter=',')

corpus = documents['text'].tolist()

tokenized_corpus = [doc.split() for doc in documents]

ranker = BM25Okapi(tokenized_corpus)


@app.route("/")
def index_page():
    return render_template("index.html")


@app.route('/<search_string>')
def search_page(search_string):
    print(search_string)

    # Rank documents based on the search string
    results = ranker.get_top_n(search_string, tokenized_corpus, n=5)

    return render_template("search.html", search_string=search_string, results=results)


@app.get("/test")
def test_page():
    return render_template("test.html")


@app.errorhandler(404)
def page_not_found(e):
    return render_template("notfound.html"), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)