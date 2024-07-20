"""
This is the Main file that has to be run to start and evaluate the search engine.

This Main file contains two options. Either you set up the whole frontend to see the search engine on your browser
or you evaluate the search engine with a .txt batch file.

This can be controlled by the variables:
1. USE_FRONTEND (Boolean): if True, it starts the frontend with the search engine on your browser.
It is accessible then under http://127.0.0.1:5001. If False, it's on "Batch Evaluation mode" and evaluates the queries.txt.
    If USE_FRONTEND is False (i.e. you want to evaluate your batch file):
    2. BATCH_PATH (String): path to your batch file.
    3. OUTPUT_PATH (String): path to your output file.

Working steps:
1. Set the variables properly for your desired use case.
2. Start the database using docker (see README.MD) and wait until a connection can be established. It might take ~20 seconds.
3. Execute `python Main.py`. It might take ~30 seconds for everything to be set up.
4. If USE_FRONTEND is True: Enter http://127.0.0.1:5001 after the console tells you that the frontend is running on your browser.
"""

import time

from flask import Flask, render_template
from flask_bootstrap import Bootstrap

from ranker.QueryResult import QueryResult
from ranker.Ranker import Ranker

app = Flask(__name__)
Bootstrap(app)

# Parameters to control. See the description above for their meaning.
USE_FRONTEND = True
BATCH_PATH = "example_queries.txt"
OUTPUT_PATH = "example_queries_eval_flat_rommel_smilga_diederichs.txt"

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


def evaluate_batch(batch_path: str, output_path: str):
    """
    Evaluate a batch of queries on our search engine and provides an output file in the end.
    Because it wasn't clear for us what the output format should really be,

    Parameters:
        batch_path (str): path to the batch file
        output_path (str): path to the output file
    """
    start_time = time.time()
    with open(batch_path, "r") as f:
        file = f.readlines()
    all_queries = []
    all_docs = ranker.all_docs
    batch_result = ""

    # Step 1: filter out every query from the batch file and add them to a list.
    for line in file:
        query = line.split("\t")[1]
        if query.endswith("\n"):
            query = query.split("\n")[0]
        all_queries.append(query)

    # Step 2: rank every query and write it to the output
    for i, query in enumerate(all_queries, start=1):
        query_result = ranker.rank_query(query, all_docs)
        for j, (doc, score) in enumerate(zip(query_result.documents, query_result.scores), start=1):
            batch_result += f"{i}\t{j}\t{doc.url}\t{score}\n"
        print(f"Ranking for the query {i}: {query} completed!")

    # Step 3: save the output file
    with open(output_path, "w") as f:
        f.write(batch_result)
    end_time = time.time()
    print(f"Batch evaluation succeded! Results written to {output_path}, Time taken: {end_time - start_time} seconds")

if __name__ == '__main__':
    """
    Starts the search engine when executing Main.py. Takes 30-60 seconds to be able to use it finally.
    Make sure that the docker container is started before starting the search engine (see README.md). 
    """
    if USE_FRONTEND:
        print(f"Starts the frontend with the search engine on http://127.0.0.1:5001. Might take a bit until it works")
        app.run(host='0.0.0.0', port=5001)
    else:
        print(f"Batch evaluation mode has started")
        evaluate_batch(BATCH_PATH, OUTPUT_PATH)
