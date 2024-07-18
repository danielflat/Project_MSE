import time

from flask import Flask, render_template
from flask_bootstrap import Bootstrap

from ranker.QueryResult import QueryResult
from ranker.Ranker import Ranker

app = Flask(__name__)
Bootstrap(app)

ranker = Ranker()


@app.route("/")
def index_page():
    return render_template("index.html")


@app.route('/search/<search_string>')
def search_page(search_string):
    print(f"User query: {search_string}. Let's go!")
    start_time = time.time()
    # Rank documents based on the search string. Contains the top 100 results
    queryResult: QueryResult = ranker.rank_query(search_string, ranker.all_docs)
    end_time = time.time()
    print(f"Search time: {end_time - start_time} seconds")
    return render_template("search.html", queryResult=queryResult)


@app.get("/test")
def test_page():
    """
    Not relevant for evaluation. Just for testing purposes.
    """
    return render_template("test.html")


@app.errorhandler(404)
def page_not_found(e):
    """
    Throws an error page. Not so important.
    """
    return render_template("notfound.html"), 404


if __name__ == '__main__':
    """
    Starts the search engine when executing Main.py. Takes 30-60 seconds to be able to use it finally.
    """
    app.run(host='0.0.0.0', port=5001, debug=True)