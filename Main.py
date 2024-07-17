import time

from flask import Flask, render_template
from flask_bootstrap import Bootstrap

from ranker.QueryResult import QueryResult
from ranker.ranker import Ranker

app = Flask(__name__)
Bootstrap(app)
#
# # Get the absolute path to the current directory
# base_dir = os.path.dirname(os.path.abspath(__file__))
#
# # Construct the absolute paths to the model and tokenizer directories
# model_path = os.path.join(base_dir, 'ranker/local_model/model')
# tokenizer_path = os.path.join(base_dir, 'ranker/local_model/tokenizer')
#
# # Print paths to ensure correctness
# print(f"Current working directory: {base_dir}")
# print(f"Model path: {model_path}")
# print(f"Tokenizer path: {tokenizer_path}")
#
# # Check if paths exist
# print("Checking if paths exist...")
# print(f"Model path exists: {os.path.exists(model_path)}")
# print(f"Tokenizer path exists: {os.path.exists(tokenizer_path)}")
#
# # Example documents - in practice, fetch documents based on the search query
# documents = pd.read_csv('dummyindex.csv', delimiter=',')
#
# corpus = documents['text'].tolist()
#
# tokenized_corpus = [doc.split() for doc in documents]
#
# ranker = BM25Okapi(tokenized_corpus)

ranker_flat = Ranker()


@app.route("/")
def index_page():
    return render_template("index.html")


@app.route('/search/<search_string>')
def search_page(search_string):
    # Sometimes there are bad queries where we only redirect to the index.html
    # if search_string is None or search_string == '' or search_string == "favicon.ico":
    #     return redirect(url_for("index_page"))

    print(f"User query: {search_string}. Let's go!")
    start_time = time.time()
    # Rank documents based on the search string. Contains the top 100 results
    queryResult: QueryResult = ranker_flat.rank_query(search_string, ranker_flat.all_docs)
    end_time = time.time()
    print(f"Search time: {end_time - start_time} seconds")
    return render_template("search.html", queryResult=queryResult)


@app.get("/test")
def test_page():
    return render_template("test.html")


@app.errorhandler(404)
def page_not_found(e):
    return render_template("notfound.html"), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)