from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from ranker.ranker import Ranker
import os


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
documents = open('dummyindex.csv').read().splitlines()


# Initialize Ranker with absolute paths
ranker = Ranker(model_path=model_path, tokenizer_path=tokenizer_path, device='cpu')


@app.route("/")
def index_page():
    return render_template("index.html")

@app.route('/search')
def search_page():
    search_string = request.args.get('q', '')
    print(search_string)

    # Rank documents based on the search string
    ranked_documents, scores = ranker.rank_documents(search_string, documents)

    results = [{"text": doc, "score": score} for doc, score in zip(ranked_documents, scores)]

    # Only return search results with good scores
    # results = [result for result in results if result['score'] > 0.2]
    print(results)

    if len(results) == 0:
        results = [{"text": "No results found", "score": 0}]
        page_not_found(404)

    return render_template("search.html", search_string=search_string, results=results)

@app.get("/test")
def test_page():
    return render_template("test.html")

@app.errorhandler(404)
def page_not_found(e):
    return render_template("notfound.html"), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
