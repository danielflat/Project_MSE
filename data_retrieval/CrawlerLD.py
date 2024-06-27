#Crawl the web. You need (at least) two parameters:
	#frontier: The frontier of known URLs to crawl. You will initially populate this with your seed set of URLs and later maintain all discovered (but not yet crawled) URLs here.
	#index: The location of the local index storing the discovered documents.
import urllib.request
from bs4 import BeautifulSoup
import re
import time
import threading
import urllib.parse
import robotexclusionrulesparser as rerp
import urllib.request
from bs4 import BeautifulSoup
import re
import time
import urllib.parse
import robotexclusionrulesparser as rerp
from queue import Queue

class Crawler:
    def __init__(self, frontier_de, max_pages):
        self.frontier_de = frontier_de 
        self.max_pages = max_pages
        self.visited = set()
        self.to_visit = Queue()
        self.robot_parsers = {}

        #Fill Queue with frontier_de
        for url in frontier_de:
            self.to_visit.put(url)

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
        for link in soup.find_all('a', href=True):
            href = link['href']
            if not href.startswith('http'):
                href = urllib.parse.urljoin(base_url, href)  # Convert relative URLs to absolute
            links.append(href)
        return links

    def index_page(self, url, html):
        """Placeholder function for indexing a page's content."""
        print(f"Indexing {url}")

    def crawl(self):
        """Main function to start the crawling process."""
        pages_crawled = 0
        while self.to_visit and pages_crawled < self.max_pages:
            url = self.to_visit.pop(0)  # Get the next URL from the frontier
            if url in self.visited or not self.is_allowed(url):
                continue
            
            print(f"Crawling: {url}")
            html = self.fetch_page(url)
            if html:
                self.visited.add(url)  # Mark the URL as visited
                self.index_page(url, html)  # Index the page
                links = self.parse_links(html, url)  # Extract links from the page
                for link in links:
                    if link not in self.visited and link not in self.to_visit:
                        self.to_visit.append(link)  # Add new links to the frontier
                pages_crawled += 1
            time.sleep(1)  # Be polite and avoid hitting the server too hard

if __name__ == "__main__":
    frontier_de = [
        'https://rp.baden-wuerttemberg.de/rpt/'
    ]
    max_pages = 100 
    crawler = Crawler(frontier_de, max_pages)
    crawler.crawl()
