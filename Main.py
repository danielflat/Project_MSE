from flask import Flask
from flask import render_template
from flask_bootstrap import Bootstrap

app = Flask(__name__)
Bootstrap(app)

@app.route("/")
def index_page():
    return render_template("index.html")


@app.route('/<search_string>')
def search_page(search_string):
    print(search_string)
    # From here on the fun is starting

    # First test implementation
    results = ['https://uni-tuebingen.de/en/',
            'https://www.tuebingen.mpg.de/en',
            'https://www.tuebingen.de/en/',
            'https://health-nlp.com/people/carsten.html'
            'https://allevents.in/tubingen/food-drinks',
            'https://www.dzne.de/en/about-us/sites/tuebingen']

    return render_template("search.html", search_string=search_string,
                           results=results)

@app.get("/test")
def test_page():
    return render_template("test.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
