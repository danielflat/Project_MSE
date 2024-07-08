# Crawl the web. You need (at least) two parameters:
# frontier: The frontier of known URLs to crawl. You will initially populate this with your seed set of URLs and later maintain all discovered (but not yet crawled) URLs here.
# index: The location of the local index storing the discovered documents.

import json
import urllib.request
from bs4 import BeautifulSoup
import time
import urllib.parse
from queue import Queue
from collections import deque 
import re
import robotexclusionrulesparser as rerp
import nltk
import en_core_web_sm
from datetime import datetime
from keybert import KeyBERT

nltk.download("stopwords")
nlp = en_core_web_sm.load()
kw_model = KeyBERT("distilbert-base-nli-mean-tokens")

# domains that are in English but are falsely detected as German
FALSE_POSITIVE_LINKS = [
    "https://www.medizin.uni-tuebingen.de/en-de/",
    "https://www.neurochirurgie-tuebingen.de/en/"
    ]

class Crawler:
    def __init__(
        self,
        frontier,
        max_pages,
        max_steps_per_domain_general,
        max_steps_per_domain_prioritised,
        timeout,
        visited=None,
        to_visit=None,
        to_visit_prioritised=None,
        visited_domains=None,
        domain_steps=None,
        extra_links=None,
        verbose=False
    ):
        self.frontier = frontier
        self.n_crawled_pages = 0
        self.max_pages = max_pages
        self.max_steps_per_domain_general = max_steps_per_domain_general
        self.max_steps_per_domain_prioritised = max_steps_per_domain_prioritised
        self.timeout = timeout
        self.visited = visited or set()
        self.to_visit = to_visit or deque()
        self.to_visit_prioritised = to_visit_prioritised or deque()
        self.robot_parsers = {}
        self.domain_steps = domain_steps or {}
        self.visited_domains = visited_domains or set()
        self.verbose = verbose
        self.extra_links = extra_links or []
        # uncomment if we want to get rid of iterable logic
        # self.scraped_webpages_info = []

    def get_robot_parser(self, url):
        """Fetches and parses the robots.txt file for the given URL's domain."""
        domain = urllib.parse.urlparse(url).netloc
        if domain in self.robot_parsers:
            return self.robot_parsers[domain]

        robots_url = urllib.parse.urljoin(f"http://{domain}", "/robots.txt")
        try:
            response = urllib.request.urlopen(robots_url, timeout=self.timeout)
            robots_txt = response.read().decode("utf-8")
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
            return parser.is_allowed("*", url)
        return True

    def fetch_page(self, url):
        """Fetches the content of a URL."""
        try:
            response = urllib.request.urlopen(url, timeout=self.timeout)
            return response.read()
        except Exception as e:
            print(f"{url}: Failed to fetch page: {e}")
            return None

    def is_media_link(self, link):
        media_extensions = {
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg', # Image extensions
            '.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm', # Video extensions
            '.mp3', '.wav', '.aac', '.flac', '.ogg', '.wma', '.m4a' # Audio extensions
            }
        return any(link.lower().endswith(ext) for ext in media_extensions)

    def parse_links(self, html, base_url):
        """Parses and returns all links found in the HTML content."""
        soup = BeautifulSoup(html, "html.parser")
        internal_links = []
        external_links = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            # to filter out links leading to other parts of the page
            # and links to images, videos, audios
            if "#" in href or self.is_media_link(href):
                continue
            # to find relative links, e.g. /menu
            if not href.startswith("http"):
                # filtering out stuff like that is also located in <a>
                if "/" not in href:
                    if self.verbose:
                        print(f"Filtered out invalid internal link: {href}")
                    continue
                href = urllib.parse.urljoin(base_url, href)
                internal_links.append(href)
            else:
                external_links.append(href)
        return internal_links, external_links

    def preprocess_text(self, text):
        # TODO hyphen is not treated the right way (e.g. we get baden and württemberg instead of baden-württemberg)
        """Lowercases, tokenizes, and removes stopwords from the page content."""
        eng_stopwords = nltk.corpus.stopwords.words("english")
        processed_text = " ".join([i for i in nltk.word_tokenize(text.lower()) if i not in eng_stopwords])
        return processed_text

    def get_keywords_with_KeyBERT(self, text, keyphrase_ngram_range=(1, 1), top_n=100):
        # TODO think of getting summaries instead of keywords; seems more similar to our train data
        """Extracts keywords from text using KeyBERT."""
        keybert_keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=keyphrase_ngram_range,
            stop_words="english",
            top_n=top_n,
        )
        keybert_keywords = [kw[0] for kw in keybert_keywords]
        # Further filter keywords to focus on nouns and proper nouns
        # TODO maybe remove this part
        filtered_keywords = [
            word for word in keybert_keywords if any(token.pos_ in ["NOUN", "PROPN"] for token in nlp(word))
        ]
        return filtered_keywords

    # NOT USED NOW, INFO NOT PRESENT IN MOST PAGES
    def get_creation_or_update_timestamp(self, soup):
        """Gets the date when the page was last updated. If not present, gets the date when the page was created."""
        date = None
        potential_selectors = [
            {"tag": "time", "attrs": {"class": "last-updated"}},
            {"tag": "div", "attrs": {"id": "last-updated"}},
            {"tag": "span", "attrs": {"class": "update-time"}},
        ]
        for selector in potential_selectors:
            last_update_tag = soup.find(selector["tag"], selector["attrs"])
            if last_update_tag:
                date = last_update_tag.get_text()

        if not date:
            date_meta = soup.find("meta", attrs={"name": "date"})
            if date_meta:
                date = date_meta.get("content")

        return date

    def index_page(self, url, html):
        # TODO also think about meta descriptions? might be useful; google uses them
        """Indexes the page content, extracting url, title, headings, page_text,
        keywords, timestamp of the page creation / last update,
        timestamp of when the crawler accessed the page."""
        webpage_content = {}
        if self.verbose:
            print(f"Indexing...")
        try:
            accessed_timestamp = datetime.now()
            soup = BeautifulSoup(html, "html.parser")
            # this line is for debugging
            # print("WEBPAGE: ", soup.prettify())
            title = soup.title.string
            page_text = " ".join(soup.get_text(separator=" ").split())
            processed_page_text = self.preprocess_text(page_text)
            keywords = self.get_keywords_with_KeyBERT(processed_page_text, keyphrase_ngram_range=(1, 1), top_n=100)
            headings = [heading.text.strip() for heading in soup.find_all(re.compile("^h[1-6]$"))]
            # created_or_updated_timestamp = self.get_creation_or_update_timestamp(soup)
            webpage_content = {
                "url": url,
                "title": title,
                "headings": headings,
                "page_text": page_text,
                "raw_html": html,
                "keywords": keywords,
                # "created_or_updated_timestamp": created_or_updated_timestamp,
                "accessed_timestamp": accessed_timestamp,
            }
            if self.verbose:
                print(f"Indexed url {url} with title `{title}` successfully.")
            # this line is for debugging
            # print(webpage_content)
        except Exception as e:
            print(f"{url}: faced an error while indexing, {e}. Moving on to the next page.")

        return webpage_content

    def is_english(self, html):
        """Checks if the HTML content is in English."""
        soup = BeautifulSoup(html, "html.parser")
        html_tag = soup.find("html")
        if html_tag and html_tag.get("lang"):
            return html_tag.get("lang").startswith("en")
        return False


    def check_if_should_be_crawled(self, domain, url):
        _should_be_crawled = True

        if url in self.visited:
            if self.verbose:
                print(f"{url}: already visited.")
            _should_be_crawled = False

        elif not self.is_allowed(url):
            if self.verbose:
                print(f"{url}: crawling not allowed.")
            _should_be_crawled = False
        
        elif self.is_media_link(url):
            if self.verbose:
                print(f"{url}: link to media.")
            _should_be_crawled = False
        
        elif domain in self.visited_domains:
            if self.verbose:
                print(f"{url}: domain hit the max number of visits.")
            _should_be_crawled = False
            self.extra_links.append(url)
        
        return _should_be_crawled

    def crawl_and_index_link(self, to_visit_queue, max_steps_per_domain):
        if self.verbose:
            print(f"Pages crawled: {self.n_crawled_pages}. \
Pages left: {self.max_pages - self.n_crawled_pages}.")
        url = to_visit_queue.popleft()
        domain = urllib.parse.urlparse(url).netloc
        webpage_info = None

        if self.check_if_should_be_crawled(domain, url):
            if self.verbose:
                print(f"Crawling: {url}")
            # add to visited even if not english / no html to avoid checking again
            self.visited.add(url)
            html = self.fetch_page(url)
            if html:
                # some links from the prioritised queue have "de" set
                # even though they are in English. we want to crawl them anyways
                if not self.is_english(html) and not any(url.startswith(fp) for fp in FALSE_POSITIVE_LINKS):
                    if self.verbose:
                        print(f"{url}: not in English.")
                else:
                    webpage_info = self.index_page(url, html)
                    internal_links, external_links = self.parse_links(html, url)

                    if domain not in self.domain_steps:
                        self.domain_steps[domain] = 1
                    else:
                        self.domain_steps[domain] += 1

                    for link in internal_links:
                        # not adding visited links; may add visited domains
                        if link not in self.visited and link not in to_visit_queue:
                            # internal links that are children of a Tuebingen-focused link 
                            # are also assumed to be Tuebingen-focused;
                            # thus they go into prioritised queue for Tuebingen-focused link
                            # and general queue for general links
                            to_visit_queue.append(link)

                    for link in external_links:
                        if link not in self.visited and link not in self.to_visit:
                            # external links are assumed to be not Tuebingen-focused;
                            # thus they always go into general (not prioritised) queue
                            self.to_visit.append(link)

                    if webpage_info:
                        webpage_info["internal_links"] = internal_links
                        webpage_info["external_links"] = external_links

                    self.n_crawled_pages += 1
                    time.sleep(1)

                    # if we visited enough pages in a given domain, add this domain to visited_domains
                    # next time a page from this domain will be skipped
                    if self.domain_steps[domain] == max_steps_per_domain:
                        self.visited_domains.add(domain)

        return webpage_info


    def __iter__(self):
        """Main function to start the crawling process."""

        if self.visited:
            print(f"""Continue crawling from a checkpoint. to_visit_prioritised len: \
{len(self.to_visit_prioritised)}, to_visit len: {len(self.to_visit)}, \
visited len: {len(self.visited)}, visited_domains len: {len(self.visited_domains)} extra_links len: {len(self.extra_links)} \
domain_steps: {self.domain_steps},
For specific values please refer to the backup json file.""")
        else:
            print("Start crawling from the very beginning. Good luck!")

        # fill queues with values from frontiers if they are not pre-defined from a checkpoint
        if not self.to_visit_prioritised:
            for url in self.frontier["tuebingen_focused_pages"]:
                self.to_visit_prioritised.append(url)
        if not self.to_visit:
            for url in self.frontier["general_pages"]:
                self.to_visit.append(url)

        # crawl Tuebingen-focused websites and their internal links first
        while self.to_visit_prioritised and self.n_crawled_pages < self.max_pages:
            try:
                webpage_info = self.crawl_and_index_link(to_visit_queue=self.to_visit_prioritised, max_steps_per_domain=self.max_steps_per_domain_prioritised)
                if webpage_info:
                    yield webpage_info, self.to_visit_prioritised, self.to_visit, self.visited_domains, self.visited, self.domain_steps, self.extra_links
            except Exception as e:
                print(f" {url}: Encountered error {e}. Skipping this url.")

        if not self.to_visit_prioritised:
            print("Finished crawling prioritised sites and their children.")

        # crawl general websites & Tuebingen-focused websites' external links
        while self.to_visit and self.n_crawled_pages < self.max_pages:
            try:
                webpage_info = self.crawl_and_index_link(to_visit_queue=self.to_visit, max_steps_per_domain=self.max_steps_per_domain_general)
                if webpage_info:
                    yield webpage_info, self.to_visit_prioritised, self.to_visit, self.visited_domains, self.visited, self.domain_steps, self.extra_links
            except Exception as e:
                print(f"{url}: Encountered error {e}. Skipping this url.")

        if not self.to_visit:
            print("Finished crawling general sites.")
        elif self.n_crawled_pages >= self.max_pages:
            print("Reached the maximum number of pages to crawl.")
        else:
            print("Something went wrong. Please double-check your code.")


if __name__ == "__main__":
    # TODO expand the frontier
    with open("../frontier.json", "r") as file:
        frontier = json.load(file)
    max_pages = 30
    max_steps_per_domain = 5
    timeout = 5  # seconds
    scraped_webpages_info = []

    crawler = Crawler(frontier, max_pages, max_steps_per_domain, timeout)
    for webpage_info in crawler:
        scraped_webpages_info = scraped_webpages_info.append(webpage_info)
