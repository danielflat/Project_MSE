#Crawl the web. You need (at least) two parameters:
	#frontier: The frontier of known URLs to crawl. You will initially populate this with your seed set of URLs and later maintain all discovered (but not yet crawled) URLs here.
	#index: The location of the local index storing the discovered documents.
import re
import urllib.request
from bs4 import BeautifulSoup
import time
import urllib.parse
import robotexclusionrulesparser as rerp
import nltk
nltk.download('stopwords')
import en_core_web_sm
from datetime import datetime
from keybert import KeyBERT
nlp = en_core_web_sm.load()
kw_model = KeyBERT('distilbert-base-nli-mean-tokens')


class Crawler:
    def __init__(self, frontier_de, max_pages):
        self.frontier_de = frontier_de 
        self.max_pages = max_pages
        self.visited = set()
        self.to_visit = list(frontier_de)
        self.robot_parsers = {}


    def get_robot_parser(self, url):
        """Fetches and parses the robots.txt file for the given URL's domain."""
        domain = urllib.parse.urlparse(url).netloc
        if domain in self.robot_parsers:
            return self.robot_parsers[domain]
        
        robots_url = urllib.parse.urljoin(f"http://{domain}", '/robots.txt')
        try:
            response = urllib.request.urlopen(robots_url)
            robots_txt = response.read().decode('utf-8')
            parser = rerp.RobotExclusionRulesParser()
            parser.parse(robots_txt)
            self.robot_parsers[domain] = parser
            return parser
        except Exception as e:
            print(f"Failed to fetch robots.txt for {domain}: {e}")
            return None

    def is_allowed(self, url):
        """Checks if a URL is allowed to be crawled based on robots.txt rules."""
        parser = self.get_robot_parser(url)
        if parser:
            return parser.is_allowed('*', url)
        return True

    def fetch_page(self, url):
        """Fetches the content of a URL."""
        try:
            response = urllib.request.urlopen(url)
            return response.read()
        except Exception as e:
            print(f"Failed to fetch page: {e}")
            return None

    def parse_links(self, html, base_url):
        """Parses and returns all links found in the HTML content."""
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        internal_links = []
        external_links = [] #maybe better to diversify quicker
        for link in soup.find_all('a', href=True):
            href = link['href']
            if not href.startswith('http'):
                href = urllib.parse.urljoin(base_url, href) 
            links.append(href)
        return links
    
    def preprocess_text(self, text):
        """Lowercases, tokenizes, and removes stopwords from the page content."""
        eng_stopwords = nltk.corpus.stopwords.words('english')
        processed_text = ' '.join([i for i in nltk.word_tokenize(text.lower()) if i not in eng_stopwords])
        return processed_text
    
    def get_keywords_with_KeyBERT(self, text):
        """Extracts keywords from text using KeyBERT."""

        keybert_keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=100)
        keybert_keywords = [kw[0] for kw in keybert_keywords]
        # Further filter keywords to focus on nouns and proper nouns
        filtered_keywords = [word for word in keybert_keywords if any(token.pos_ in ['NOUN', 'PROPN'] for token in nlp(word))]
        return filtered_keywords

    def get_creation_or_update_timestamp(self, soup):
        """Gets the date when the page was last updated. If not present, gets the date when the page was created."""
        date = None
        potential_selectors = [
            {'tag': 'time', 'attrs': {'class': 'last-updated'}},
            {'tag': 'div', 'attrs': {'id': 'last-updated'}},
            {'tag': 'span', 'attrs': {'class': 'update-time'}},
        ]
        for selector in potential_selectors:
            last_update_tag = soup.find(selector['tag'], selector['attrs'])
            if last_update_tag:
                date = last_update_tag.get_text()

        if not date:
            date_meta = soup.find('meta', attrs={'name': 'date'})
            if date_meta:
                date = date_meta.get('content')

        return date
            
    def index_page(self, url, html):
        # TODO also think about meta descriptions? might be useful; google uses them
        """Indexes the page content, extracting url, title, headings, page_text,
        keywords, timestamp of the page creation / last update,
        timestamp of when the crawler accessed the page."""
        webpage_content = {}
        print(f"Indexing...")
        try:
            accessed_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            soup = BeautifulSoup(html, 'html.parser')
            # this line is for debugging
            # print("WEBPAGE: ", soup.prettify())
            title = soup.title.string
            page_text = ' '.join(soup.get_text(separator=' ').split())
            page_text = self.preprocess_text(page_text)
            keywords = self.get_keywords_with_KeyBERT(page_text)
            headings = [heading.text.strip() for heading in soup.find_all(re.compile('^h[1-6]$'))]
            created_or_updated_timestamp = self.get_creation_or_update_timestamp(soup)
            webpage_content = {
                "url": url,
                "title": title,
                "headings": headings,
                "page_text": page_text,
                "keywords": keywords,
                "created_or_updated_timestamp": created_or_updated_timestamp,
                "accessed_timestamp": accessed_timestamp,
            }
            print(f"Indexed url {url} with title `{title}` successfully.")
        except Exception as e:
            print(f"Faced an error while indexing url {url}: {e}. Moving on to the next page.")

        return webpage_content

    def crawl(self):
        """Main function to start the crawling process."""
        pages_crawled = 0
        while self.to_visit and pages_crawled < self.max_pages:
            url = self.to_visit.pop(0) 
            if url in self.visited or not self.is_allowed(url):
                continue
            
            print(f"Crawling: {url}")
            html = self.fetch_page(url)
            if html:
                self.visited.add(url)  
                webpage_info = self.index_page(url, html)
                links = self.parse_links(html, url)  
                for link in links:
                    if link not in self.visited and link not in self.to_visit:
                        self.to_visit.append(link)  
                pages_crawled += 1
            time.sleep(1)

if __name__ == "__main__":
    frontier_de = [
        'https://uni-tuebingen.de/en/',
        # 'https://www.tuebingen.mpg.de/en',
        # 'https://www.tuebingen.de/en/',
        # 'https://health-nlp.com/people/carsten.html',
        # 'https://www.dzne.de/en/about-us/sites/tuebingen',
        # 'https://www.britannica.com/place/Tubingen-Germany',
        # 'https://tuebingenresearchcampus.com/en/tuebingen/general-information/local-infos/',
        # 'https://wanderlog.com/list/geoCategory/199488/where-to-eat-best-restaurants-in-tubingen',
        # 'https://wikitravel.org/en/T%C3%BCbingen'
    ]
    max_pages = 1

    crawler = Crawler(frontier_de, max_pages)
    crawler.crawl()
