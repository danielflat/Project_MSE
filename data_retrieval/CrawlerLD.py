#Crawl the web. You need (at least) two parameters:
	#frontier: The frontier of known URLs to crawl. You will initially populate this with your seed set of URLs and later maintain all discovered (but not yet crawled) URLs here.
	#index: The location of the local index storing the discovered documents.
import threading
import urllib.request
from bs4 import BeautifulSoup
import time
import urllib.parse
from queue import Queue
import robotexclusionrulesparser as rerp

class Crawler:
    def __init__(self, frontier_de, max_pages):
        self.frontier_de = frontier_de 
        self.max_pages = max_pages
        self.visited = set()
        self.to_visit = list(frontier_de                    )
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

    def index_page(self, url, html):
        """Placeholder function for indexing a page's content."""
        print(f"Indexing {url}")

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
                self.index_page(url, html) 
                links = self.parse_links(html, url)  
                for link in links:
                    if link not in self.visited and link not in self.to_visit:
                        self.to_visit.append(link)  
                pages_crawled += 1
            time.sleep(1)  

if __name__ == "__main__":
    frontier_de = [
        'https://uni-tuebingen.de/en/',
        'https://www.tuebingen.mpg.de/en',
        'https://www.tuebingen.de/en/',
        'https://health-nlp.com/people/carsten.html',
        'https://www.dzne.de/en/about-us/sites/tuebingen',
        'https://www.britannica.com/place/Tubingen-Germany',
        'https://tuebingenresearchcampus.com/en/tuebingen/general-information/local-infos/',
        'https://wanderlog.com/list/geoCategory/199488/where-to-eat-best-restaurants-in-tubingen',
        'https://wikitravel.org/en/T%C3%BCbingen'
    ]
    max_pages = 100 

    crawler = Crawler(frontier_de, max_pages)
    crawler.crawl()
