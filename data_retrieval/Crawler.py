# Crawl the web. You need (at least) two parameters: frontier: The frontier of known URLs to crawl. You will
# initially populate this with your seed set of URLs and later maintain all discovered (but not yet crawled) URLs
# here. index: The location of the local index storing the discovered documents.

import json
import urllib.request
from bs4 import BeautifulSoup
import time
import urllib.parse
from queue import Queue
import re
import robotexclusionrulesparser as rerp
import nltk
import en_core_web_sm  # df: Make sure to run "python -m spacy download en_core_web_sm" for this import to work
from datetime import datetime
from keybert import KeyBERT
from nltk.corpus import stopwords
from urllib.parse import urlparse

from utils.directoryutil import get_path

NLTK_PATH = get_path("nltk_data")

nltk.data.path.append(NLTK_PATH)
nltk.download("stopwords", download_dir=NLTK_PATH)
nltk.download("punkt", download_dir=NLTK_PATH)
nlp = en_core_web_sm.load()  # df: I am not sure if we are allowed to use this one
kw_model = KeyBERT("distilbert-base-nli-mean-tokens")


class Crawler:
    def __init__(self, frontier, max_pages, max_steps_per_domain, timeout):
        self.frontier = frontier
        self.n_crawled_pages = 0
        self.max_pages = max_pages
        self.max_steps_per_domain = max_steps_per_domain
        self.timeout = timeout
        self.visited = set()
        self.to_visit = Queue()
        self.to_visit_prioritised = Queue()
        for url in frontier["tuebingen_focused_pages"]:
            self.to_visit_prioritised.put(url)
        for url in frontier["general_pages"]:
            self.to_visit.put(url)
        self.robot_parsers = {}
        self.domain_steps = {}
        self.visited_domains = set()
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
            print(f"Failed to fetch page: {e}")
            return None

    def is_internal_link(self, base_url, link_url):
        """
        Determines if `link_url` is an internal link of the base domain.
        Returns: True if it is, False otherwise.
        """
        return (not link_url.startswith("http") # "https" is included here; to find relative links, e.g. /menu
                or self.is_subdomain(base_url, link_url))

    def is_subdomain(self, base_url, test_url):
        """
        Finds out if the url to test is a subdomain of the original page. Subdomains are getting treated as
        `internal links`
        Returns: True if it is, False otherwise.
        """

        # Parse the URLs
        main_domain = urlparse(base_url).hostname
        test_domain = urlparse(test_url).hostname

        # Check if test_url ends with base_url
        return test_domain.endswith(main_domain)

    def parse_links(self, html, base_url):
        """Parses and returns all links found in the HTML content."""
        soup = BeautifulSoup(html, "html.parser")
        internal_links = []
        external_links = []
        for a_tag in soup.find_all("a", href=True):
            link_url = a_tag["href"]
            if "#" in link_url:
                continue
            if self.is_internal_link(base_url, link_url):
                # filtering out stuff like that is also located in <a>
                if "/" not in link_url:
                    print(f"Filtered out invalid internal link: {link_url}")
                    continue
                link_url = urllib.parse.urljoin(base_url, link_url)
                internal_links.append(link_url)
            else:
                external_links.append(link_url)
        return internal_links, external_links

    def preprocess_text(self, text):
        # TODO hyphen is not treated the right way (e.g. we get baden and württemberg instead of baden-württemberg)
        """Lowercases, tokenizes, and removes stopwords from the page content."""
        eng_stopwords = set(stopwords.words("english"))
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
                "keywords": keywords,
                # "created_or_updated_timestamp": created_or_updated_timestamp,
                "accessed_timestamp": accessed_timestamp,
            }
            print(f"Indexed url {url} with title `{title}` successfully.")
            # this line is for debugging
            # print(webpage_content)
        except Exception as e:
            print(f"Faced an error while indexing url {url}: {e}. Moving on to the next page.")

        return webpage_content

    def is_english(self, html):
        """Checks if the HTML content is in English."""
        soup = BeautifulSoup(html, "html.parser")
        html_tag = soup.find("html")
        if html_tag and html_tag.get("lang"):
            return html_tag.get("lang").startswith("en")
        return False

    def crawl_and_index_prioritised_link(self):
        print(f"Pages crawled: {self.n_crawled_pages}. Pages left: {self.max_pages - self.n_crawled_pages}.")
        url = self.to_visit_prioritised.get()
        domain = urllib.parse.urlparse(url).netloc
        webpage_info = None

        # no visited domain restriction here
        if url in self.visited or not self.is_allowed(url):
            return webpage_info

        print(f"Crawling: {url}")
        # add to visited even if not english / no html to avoid checking again
        self.visited.add(url)
        self.visited_domains.add(domain)

        html = self.fetch_page(url)
        if html and self.is_english(html):
            webpage_info = self.index_page(url, html)
            internal_links, external_links = self.parse_links(html, url)

            for link in internal_links:
                # for internal links on Tuebingen-focused websites, there is no limit on domain steps
                # internal links that are children of a prioritised link are also prioritised
                self.to_visit_prioritised.put(link)

            for link in external_links:
                if link not in self.visited and link not in self.visited_domains:
                    # for unknown links, assume that they are not Tuebingen-focused 
                    # and put them to general (not prioritised) queue
                    self.to_visit.put(link)

            if webpage_info:
                webpage_info["internal_links"] = internal_links
                webpage_info["external_links"] = external_links

            self.n_crawled_pages += 1
            time.sleep(1)
        return webpage_info

    def crawl_and_index_general_link(self):
        print(f"Pages crawled: {self.n_crawled_pages}. Pages left: {self.max_pages - self.n_crawled_pages}.")
        url = self.to_visit.get()
        domain = urllib.parse.urlparse(url).netloc
        webpage_info = None

        if url in self.visited or domain in self.visited_domains or not self.is_allowed(url):
            return webpage_info

        print(f"Crawling: {url}")
        # add to visited even if not english / no html to avoid checking again
        self.visited.add(url)
        html = self.fetch_page(url)
        if html and self.is_english(html):
            webpage_info = self.index_page(url, html)
            internal_links, external_links = self.parse_links(html, url)

            if domain not in self.domain_steps:
                self.domain_steps[domain] = 0

            for link in internal_links:
                # break if more than max_steps_per_domain internal links are already in to_visit queue
                if self.domain_steps[domain] < self.max_steps_per_domain:
                    self.to_visit.put(link)
                    self.domain_steps[domain] += 1
                else:
                    break

            for link in external_links:
                external_link_domain = urllib.parse.urlparse(link).netloc
                if link not in self.visited and external_link_domain not in self.visited_domains:
                    self.to_visit.put(link)

            if webpage_info:
                webpage_info["internal_links"] = internal_links
                webpage_info["external_links"] = external_links

            self.n_crawled_pages += 1
            time.sleep(1)
        return webpage_info

    def __iter__(self):
        """Main function to start the crawling process."""

        # crawl Tübingen-focused websites and their internal links first
        while not self.to_visit_prioritised.empty() and self.n_crawled_pages < self.max_pages:
            webpage_info = self.crawl_and_index_prioritised_link()
            if webpage_info:
                yield webpage_info

        if self.to_visit_prioritised.empty():
            print("Finished crawling prioritised sites and their children.")

        # crawl general websites & Tübingen-focused websites' external links
        while not self.to_visit.empty() and self.n_crawled_pages < self.max_pages:
            webpage_info = self.crawl_and_index_general_link()
            if webpage_info:
                yield webpage_info

        if self.to_visit.empty():
            print("Finished crawling all sites; queue is empty.")
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
