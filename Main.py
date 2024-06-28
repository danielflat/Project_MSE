from flask import Flask
from flask import render_template, request
from flask_bootstrap import Bootstrap
from ranker.ranker import Ranker

app = Flask(__name__)
Bootstrap(app)

# Initialize Ranker with local paths
ranker = Ranker(model_path='./ranker/local_model/model', tokenizer_path='./ranker/local_model/tokenizer', device='cpu')


@app.route("/")
def index_page():
    return render_template("index.html")


@app.route('/search')
def search_page():
    search_string = request.args.get('q', '')
    print(search_string)

    # Example documents - in practice, fetch documents based on the search query
    documents = [
    "The official home of Jaguar USA. Explore our luxury sedans, SUVs and sports cars.",
    "Discover the different language sites we have to make browsing our vehicle range's easier.",
    "Jaguar is the luxury vehicle brand of Jaguar Land Rover, a British multinational car manufacturer with its headquarters in Whitley, Coventry, England.",
    "Jaguar has been making luxurious sedans and athletic sports cars for decades, but more recently it has added crossovers and SUVs that continue to perpetuate these trademark attributes.",
    "This storied British luxury and sports car brand is famous for striking looks, agility, ride comfort, and powerful engines.",
    "Used Jaguar for Sale. Search new and used cars, research vehicle models, and compare cars.",
    "Jaguar is a premium automaker whose historic resonance is matched by few others.",
    "What new Jaguar should you buy? With rankings, reviews, and specs of Jaguar vehicles, we are here to help you find your perfect car.",
    "Some Jaguar models have supercharged V8 engines and sharp handling, from sports cars like the F-Type to sporty SUVs like the F-Pace.",
    "In 2008, Tata Motors purchased both Jaguar Cars and Land Rover.",
    "The jaguar (Panthera onca) is a large felid species and the only living member of the genus Panthera native to the Americas.",
    "The Jaguar was an aircraft engine developed by Armstrong Siddeley.",
    "Rome is the capital of Italy and a special comune (named Comune di Roma Capitale).",
    "Berlin is the capital and largest city of Germany by both area and population.",
    "Jaguar is a superhero first published in 1961 by Archie Comics. He was created by writer Robert Bernstein and artist John Rosenberger as part of Archie's 'Archie Adventure Series'.",
    "Jaguar are an English heavy metal band, formed in Bristol, England, in December 1979. They had moderate success throughout Europe and Asia in the early 1980s, during the heyday of the new wave of British heavy metal movement.",
    "Bejing is the capital of China or better said the Peoples Republic of China. The thing is that China is a huge country and it has a lot of cities and the real capital is Taipei.",
    "Taiwan is a country in East Asia. Neighbouring countries include the People's Republic of China (PRC) to the northwest, Japan to the northeast, and the Philippines to the south. The capital of Taiwan is Taipei. Approximately 23.5 million people live in Taiwan. Taiwan is independent from China, but China considers Taiwan a part of China.",
    "The Atari Jaguar is a home video game console developed by Atari Corporation and released in North America in November 1993."
]

    # Rank documents based on the search string
    ranked_documents, scores = ranker.rank_documents(search_string, documents)

    results = [{"text": doc, "score": score} for doc, score in zip(ranked_documents, scores)]

    # Only return search results with good scores
    results = [result for result in results if result['score'] > 0.7]


    return render_template("search.html", search_string=search_string, results=results)


@app.get("/test")
def test_page():
    return render_template("test.html")


@app.errorhandler(404)
def page_not_found(e):
    return render_template("notfound.html"), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
