from flaskr import Flask, request, jsonify
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT
from whoosh.qparser import QueryParser
import os

app = Flask(__name__)

# Define the schema for Whoosh
schema = Schema(title=TEXT(stored=True), content=TEXT(stored=True))

# Create index directory if it doesn't exist
if not os.path.exists("index"):
    os.mkdir("index")
    ix = create_in("index", schema)
else:
    ix = open_dir("index")

# Function to add documents to the index
def add_documents():
    writer = ix.writer()
    writer.add_document(title="Statue of Liberty", content="An iconic symbol of freedom in New York.")
    writer.add_document(title="Central Park", content="A large public park in New York City.")
    writer.commit()

# Add initial documents
add_documents()

@app.route('/search')
def search():
    query_str = request.args.get('q')
    with ix.searcher() as searcher:
        query = QueryParser("content", ix.schema).parse(query_str)
        results = searcher.search(query)
        results_list = [{"title": result['title'], "content": result['content']} for result in results]
        return jsonify(results_list)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
