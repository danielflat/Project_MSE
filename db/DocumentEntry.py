import uuid
from datetime import datetime


class DocumentEntry(object):

    def __init__(self, url: str, keywords: list[str], content: str, last_updated: datetime, id=None):
        """
        The constructor for each document entry. This can be saved in our database then
        :param url: The string of the url
        :param keywords: The list of keywords that represents this document. Each keyword should be a string as well
        :param content: The HTML content of the document
        :param last_updated: The timestamp of the document. This can be read up from the HTML headers
        """

        self.id = id or uuid.uuid4()
        self.url = url
        self.keywords = keywords
        self.content = content
        self.last_updated = last_updated

    def __str__(self):
        return str(self.id), self.url, self.keywords, self.content, self.last_updated
    def __repr__(self):
        """
        Represents the object as a short string
        """
        id_display = self.id[:10] + '...' if len(self.id) > 10 else self.id
        keywords_display = "['" + self.keywords[0].__str__() + "', '" + self.keywords[1] + "', ...]" if len(
            self.keywords) > 2 else str(self.keywords)
        content_display = self.content[:10] + '...' if len(self.content) > 10 else self.content

        return (f"DocumentEntry[id={id_display}, "
                f"url={self.url}, "
                f"keywords={keywords_display}, "
                f"content={content_display}, "
                f"last_updated={self.last_updated}]")

    def fullString(self):
        """
        Returns the full string representation of the object. WARNING: This can be HUGE
        """
        return (f"DocumentEntry[id={self.id}, "
                f"url={self.url}, "
                f"keywords={self.keywords}, "
                f"content={self.content}, "
                f"last_updated={self.last_updated}]")
