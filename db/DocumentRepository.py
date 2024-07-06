import psycopg2
from datetime import datetime, timedelta

from db.DocumentEntry import DocumentEntry
from psycopg2.extras import execute_values


class DocumentRepository:

    def __init__(self):
        try:
            # Connect to your PostgreSQL database.
            # Here are the credentials written so the machine knows where connect to.
            self.connection = psycopg2.connect(
                user="user",
                password="user_pw",
                host="localhost",
                port="5432",
                database="search_engine_db"
            )

            # Create a cursor object. We need this to execute queries.
            self.cursor = self.connection.cursor()

            # for sanity check (SC)
            print("SC: Connected to the db. Now you can go and build the best search engine around!")
        except (Exception, psycopg2.Error) as error:
            print("SC: Connecting to PostgreSQL did not work. Maybe try to run it again.", error)

        # We save every query expression, so we have a good overview which operations we do with the database
        self.insertQuery: str = """
                                INSERT INTO documents (id, url, title, headings, page_text, keywords,
                                 accessed_timestamp, internal_links, external_links) VALUES 
                                 (%s, %s, %s, %s, %s, %s, %s, %s, %s) 
                                """
        self.insertAllQuery: str = """
                        INSERT INTO documents (id, url, title, headings, page_text, keywords, accessed_timestamp, internal_links, external_links) VALUES %s 
                        """
        self.updateQuery = """
                UPDATE documents
                SET id = %s,
                    url = %s,
                    title = %s,
                    headings = %s,
                    page_text = %s,
                    keywords = %s,
                    accessed_timestamp = %s,
                    internal_links = %s,
                    external_links = %s
                WHERE url = %s;
                """

        self.deleteQuery: str = """
        DELETE FROM documents
        WHERE url = %s;
        """

        self.deleteAllQuery: str = """
                                DELETE FROM documents
                                """

    def loadAllDocuments(self) -> list[DocumentEntry]:
        self.cursor.execute("SELECT * FROM documents")
        rows = self.cursor.fetchall()

        # Convert rows to list of TestClass objects
        result_list: list[DocumentEntry] = [DocumentEntry(url=row[1],
                                                          title=row[2],
                                                          headings=row[3],
                                                          page_text=row[4],
                                                          keywords=row[5],
                                                          accessed_timestamp=row[6],
                                                          internal_links=row[7],
                                                          external_links=row[8],
                                                          id=row[0]) for row in rows]

        return result_list

    def saveDocument(self, document: DocumentEntry):
        """
        Saves a document to the database IF the iq of the document does not exist
        """
        value: str = document.__str__()

        self.cursor.execute(self.insertQuery, value)
        self.connection.commit()
        print("SC: Saved document.")

    def updateDocument(self, document: DocumentEntry):
        """
        Updates a document of the database
        """
        values = [str(document.id), document.url, document.title, document.headings, document.page_text, document.keywords,
                document.accessed_timestamp,
                document.internal_links, document.external_links, document.url]

        self.cursor.execute(self.updateQuery, values)
        self.connection.commit()
        print("SC: Updated document.")

    def deleteDocument(self, document: DocumentEntry):
        """
        Deletes a document of the database
        """
        value = (str(document.url),)

        self.cursor.execute(self.deleteQuery, value)
        self.connection.commit()
        print("SC: Deleted document.")

    def saveAllDocuments(self, documents: list[DocumentEntry]) -> None:
        """
            Saves a list of documents to the database IF none of the iqs of the document already exist
        """
        values = [doc.__str__() for doc in documents]

        execute_values(self.cursor, self.insertAllQuery, values)
        self.connection.commit()
        print("SC: All documents saved.")

    def deleteAllDocuments(self) -> None:
        """
        Clears all documents from the database. WARNING: You cannot undo this operation.
        """
        self.cursor.execute(self.deleteAllQuery)
        self.connection.commit()
        print("SC: Deleted all documents.")
