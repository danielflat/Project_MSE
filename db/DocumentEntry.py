class DocumentEntry(object):

    def __init__(self, url: str, keywords: list[str], content: str, last_updated: str):
        """
        The constructor for each document entry. This can be saved in our database then
        :param url: The string of the url
        :param keywords: The list of keywords that represents this document. Each keyword should be a string as well
        :param content: The HTML content of the document
        :param last_updated: The timestamp of the document. This can be read up from the HTML headers
        """

        self.id = None
        self.url = url
        self.keywords = keywords
        self.content = content
        self.last_updated = last_updated
