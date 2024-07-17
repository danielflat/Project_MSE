import uuid
from datetime import datetime

import torch


class DocumentEntry(object):

    def __init__(self, url: str, title: str, headings: list[str], page_text: str, keywords: list[str],
                 accessed_timestamp: datetime, internal_links: list[str],
                 external_links: list[str], enc_text: torch.tensor = None, summary: str = None, id=None):
        """
        The constructor for each document entry. This can be saved in our database then
        :param url: The string of the url
        :param keywords: The list of keywords that represents this document. Each keyword should be a string as well
        :param content: The HTML content of the document
        :param last_updated: The timestamp of the document. This can be read up from the HTML headers
        """

        self.id = id or uuid.uuid4()
        self.url = url
        self.title = title
        self.headings = headings
        self.page_text = page_text
        self.keywords = keywords
        self.accessed_timestamp = accessed_timestamp
        self.internal_links = internal_links
        self.external_links = external_links
        self.enc_text = enc_text
        self.summary = summary

    def __str__(self):
        return (str(self.id), self.url, self.title, self.headings, self.page_text, self.keywords,
                self.accessed_timestamp, self.internal_links, self.external_links, self.enc_text.tolist(), self.summary)

    def __repr__(self):
        """
        Represents the object as a short string
        """
        id_display = str(self.id)[:10] + '...' if len(str(self.id)) > 10 else str(self.id)
        keywords_display = "['" + self.keywords[0].__str__() + "', '" + self.keywords[1] + "', ...]" if len(
            self.keywords) > 2 else str(self.keywords)
        text_display = self.page_text[:10] + '...' if len(self.page_text) > 10 else self.page_text

        return (f"DocumentEntry[id={id_display}, "
                f"url={self.url}, "
                f"title={self.title}, "
                f"headings={self.headings}, "
                f"page_text={text_display}, "
                f"keywords={keywords_display}, "
                f"accessed_timestamp={self.accessed_timestamp}, "
                f"internal_links={self.internal_links}, "
                f"external_links={self.external_links}]")

    def fullString(self):
        """
        Returns the full string representation of the object. WARNING: This can be HUGE
        """
        return (f"DocumentEntry[id={str(self.id)}, "
                f"url={self.url}, "
                f"title={self.title}, "
                f"headings={self.headings}, "
                f"page_text={self.page_text}, "
                f"keywords={self.keywords}, "
                f"accessed_timestamp={self.accessed_timestamp}, "
                f"internal_links={self.internal_links}, "
                f"external_links={self.external_links}]")
