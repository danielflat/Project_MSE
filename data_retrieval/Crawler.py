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
    def __init__(self, frontier_de, max_pages, max_steps_per_domain):
        self.frontier_de = frontier_de
        self.max_pages = max_pages
        self.max_steps_per_domain = max_steps_per_domain
        self.timeout = timeout
        self.visited = set()
        self.to_visit = Queue()
        for url in frontier_de:
            self.to_visit.put(url)
        self.robot_parsers = {}
        self.domain_steps = {}
        self.visited_domains = set()

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
            response = urllib.request.urlopen(url, timeout=self.timeout)
            return response.read()
        except Exception as e:
            print(f"Failed to fetch page: {e}")
            return None

    def parse_links(self, html, base_url):
        """Parses and returns all links found in the HTML content."""
        soup = BeautifulSoup(html, 'html.parser')
        internal_links = []
        external_links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if '#' in href:
                continue
            if not href.startswith('http'):
                href = urllib.parse.urljoin(base_url, href)
                internal_links.append(href)
            else:
                external_links.append(href)
        return internal_links, external_links


    def is_english(self, html):
        """Checks if the HTML content is in English."""
        soup = BeautifulSoup(html, 'html.parser')
        html_tag = soup.find('html')
        if html_tag and html_tag.get('lang'):
            return html_tag.get('lang').startswith('en')
        return False
    

    def index_page(self, url, html):
        """Placeholder function for indexing a page's content."""
        print(f"Indexing {url}")

    def crawl(self):
        """Main function to start the crawling process."""
        pages_crawled = 0
        while not self.to_visit.empty() and pages_crawled < self.max_pages:
            url = self.to_visit.get()
            domain = urllib.parse.urlparse(url).netloc
            if url in self.visited or domain in self.visited_domains or not self.is_allowed(url):
                continue
            
            print(f"Crawling: {url}")
            html = self.fetch_page(url)
            if html and self.is_english(html):
                self.visited.add(url)
                self.index_page(url, html)
                internal_links, external_links = self.parse_links(html, url)
                
                if domain not in self.domain_steps:
                    self.domain_steps[domain] = 0

                for link in internal_links:
                    if self.domain_steps[domain] < self.max_steps_per_domain:
                        self.to_visit.put(link)
                        self.domain_steps[domain] += 1
                    else:
                        break

                if self.domain_steps[domain] >= self.max_steps_per_domain:
                    self.visited_domains.add(domain)

                for link in external_links:
                    if link not in self.visited and link not in self.visited_domains:
                        self.to_visit.put(link)

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
            'https://wikitravel.org/en/T%C3%BCbingen',
            'https://velvetescape.com/things-to-do-in-tubingen/',
            'https://thespicyjourney.com/magical-things-to-do-in-tubingen-in-one-day-tuebingen-germany-travel-guide/',
            'https://www.outdooractive.com/en/places-to-eat-drink/tuebingen/eat-drink-in-tuebingen/21873363',
            'https://www.komoot.com/guide/210692/attractions-around-tuebingen',
            'https://bestplacesnthings.com/places-to-visit-tubingen-baden-wurttemberg-germany/',
            'https://www.citypopulation.de/en/germany/badenwurttemberg/t%C3%BCbingen/08416041__t%C3%BCbingen/',
            'https://www.braugasthoefe.de/en/guesthouses/gasthausbrauerei-neckarmueller/',
            'https://www.germany.travel/en/cities-culture/tuebingen.html',
            'https://www.tripadvisor.com/Tourism-g198539-Tubingen_Baden_Wurttemberg-Vacations.html',
            'https://www.hih-tuebingen.de/en/?no_cache=1',
            'https://tuebingenresearchcampus.com/',
            'https://cyber-valley.de/en',
            'https://www.bccn-tuebingen.de/',
            'https://www.tripadvisor.de/Attractions-g198539-Activities-Tubingen_Baden_Wurttemberg.html',
            'https://justinpluslauren.com/things-to-do-in-tubingen-germany/',
            'https://www.medizin.uni-tuebingen.de/en-de/das-klinikum',
            'https://www.medizin.uni-tuebingen.de/en-de/startseite',
            'https://bookinghealth.com/university-hospital-tuebingen',
            'https://www.eventbrite.de/d/germany--tübingen/parties/',
            'https://www.tripadvisor.de/Restaurants-g198539-c11-Tubingen_Baden_Wurttemberg.html',
            'https://www.11880.com/suche/china-restaurant/tuebingen',
            'https://de.restaurantguru.com/chinese-Tubingen-c20',
            'https://en.wikipedia.org/wiki/Neckar',
            'https://www.iksr.org/en/topics/rhine/sub-basins/neckar',
            'https://www.eventbrite.com/d/germany--tübingen/events/',
            'https://www.tripadvisor.com/Attractions-g198539-Activities-c49-Tubingen_Baden_Wurttemberg.html',
            'https://www.tripadvisor.com/Attractions-g198539-Activities-c57-Tubingen_Baden_Wurttemberg.html',
            'https://www.tuebingen.de/en/3328.html',
            'https://en.wikipedia.org/wiki/Tübingen',
            'https://www.accuweather.com/en/de/tübingen/72070/weather-forecast/167215',
            'https://www.timeanddate.com/weather/germany/tuebingen/ext',
            'https://www.tripadvisor.com/Attractions-g198539-Activities-c20-Tubingen_Baden_Wurttemberg.html',
            'https://www.yelp.com/search?cflt=bookstores&find_loc=Tübingen%2C+Baden-Württemberg',
            'https://allevents.in/tubingen',
            'https://globaltravelescapades.com/things-to-do-in-tubingen-germany/',
            'https://www.tripadvisor.com/Restaurants-g198539-Tubingen_Baden_Wurttemberg.html',
            'https://guide.michelin.com/en/de/baden-wurttemberg/tbingen/restaurants',
            'https://www.thefork.com/restaurants/tubingen-c561333',
            'https://www.medizin.uni-tuebingen.de/en-de/das-klinikum/einrichtungen/kliniken/zahn-mund-und-kieferheilkunde',
            'https://www.tripadvisor.com/Attractions-g198539-Activities-c61-Tubingen_Baden_Wurttemberg.html',
            'https://www.my-stuwe.de/en/refectory/',
            'https://www.my-stuwe.de/en/refectory/refectory-shedhalle/',
            'https://www.my-stuwe.de/en/refectory/refectory-morgenstelle-tuebingen/',
            'https://www.bahnhof.de/en/tuebingen-hbf'
    ]
    max_pages = 30
    max_steps_per_domain = 5
    timeout = 5 #seconds

    crawler = Crawler(frontier_de, max_pages, max_steps_per_domain)
    crawler.crawl()

