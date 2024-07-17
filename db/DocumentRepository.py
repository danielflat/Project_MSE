import os

import docker
import numpy as np
import psycopg2
import torch
from docker.errors import DockerException
from psycopg2.extras import execute_values
from transformers import BertTokenizer

from db.DocumentEntry import DocumentEntry
from utils.directoryutil import get_path


class DocumentRepository:

    def __init__(self):
        try:
            # Connect to your PostgreSQL database.
            # Here are the credentials written, so the machine knows where connect to.
            self.connection = psycopg2.connect(
                user="user",
                password="user_pw",
                host="localhost",
                port="5432",
                database="search_engine_db"
            )

            # Create a cursor object. We need this to execute queries.
            self.cursor = self.connection.cursor()

            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

            # for sanity check (SC)
            print("SC: Connected to the db. Now you can go and build the best search engine around!")
        except (Exception, psycopg2.Error) as error:
            print("SC: Connecting to PostgreSQL did not work. Maybe try to run it again.", error)

        # We save every query expression, so we have a good overview which operations we do with the database
        self.insertQuery: str = """
                                INSERT INTO documents (id, url, title, headings, page_text, keywords,
                                 accessed_timestamp, internal_links, external_links, enc_text, summary) VALUES 
                                 (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
                                """
        self.insertAllQuery: str = """
                        INSERT INTO documents (id, url, title, headings, page_text, keywords, accessed_timestamp, internal_links, external_links, enc_text, summary) VALUES %s 
                        """
        self.updateQuery = """
                UPDATE documents
                SET url = %s,
                    title = %s,
                    headings = %s,
                    page_text = %s,
                    keywords = %s,
                    accessed_timestamp = %s,
                    internal_links = %s,
                    external_links = %s,
                    enc_text = %s,
                    summary = %s
                WHERE id = %s;
                """

        self.deleteQuery: str = """
        DELETE FROM documents
        WHERE url = %s;
        """

        self.deleteAllQuery: str = """
                                DELETE FROM documents
                                """

    def _compute_idf(self, documents: list[np.array]):
        print("SC: Computing document IDF...")
        N = len(documents)
        idf = {}
        for document in documents:
            for word in document:
                if word in idf:
                    idf[word] += 1
                else:
                    idf[word] = 1
        print("SC:  document IDF...")
        for word, count in idf.items():
            idf[word] = np.log(((N + 1) / (count + 0.5)) + 1)
        print("SC: Finished computing document IDF...")
        return idf

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
                                                          enc_text=torch.tensor([float(d) for d in row[9]]) if row[9] is not None else None,
                                                          summary=row[10],
                                                          id=row[0]) for row in rows]

        return result_list

    def getDocumentOfUrl(self, url: str) -> DocumentEntry:
        self.cursor.execute("SELECT * FROM documents WHERE url = (%s)", (url,))
        rows = self.cursor.fetchall()

        assert len(rows) == 1
        row = rows[0]
        # Convert rows to DocumentEntry
        return DocumentEntry(url=row[1],
                             title=row[2],
                             headings=row[3],
                             page_text=row[4],
                             keywords=row[5],
                             accessed_timestamp=row[6],
                             internal_links=row[7],
                             external_links=row[8],
                             enc_text=row[9],
                             summary=row[10],
                             id=row[0])

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
        values = [document.url, document.title, document.headings, document.page_text,
                  document.keywords,
                  document.accessed_timestamp,
                  document.internal_links, document.external_links, document.enc_text.tolist(), document.summary,
                  str(document.id)]

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


    def getEncodedTextOfAllDocuments(self) -> dict[str, np.array]:
        """
        Encodes the text using BERT tokenizer
        """
        allDocuments = self.loadAllDocuments()
        documents_vectors = {}
        for doc in allDocuments:
            encode = self.tokenizer.encode(doc.page_text, add_special_tokens=False)
            doc_vec = np.array(encode)
            documents_vectors[doc.url] = doc_vec
        return documents_vectors

    def deleteAllDocuments(self) -> None:
        """
        Clears all documents from the database. WARNING: You cannot undo this operation.
        """
        self.cursor.execute(self.deleteAllQuery)
        self.connection.commit()
        print("SC: Deleted all documents.")

    def overwrite_dump(self):
        """
        Overwrites the old "./db/dump.sql" with the current state of your database-container.
        """

        try:
            # Look first if docker is running as a sanity check
            client = docker.from_env()
            client.ping()

            # Get the container_id of the db-container
            container_id = self._get_container_id_by_image("project_mse-db")

            # 1. Delete the old dump.sql
            dump_path = get_path("db/dump.sql")
            os.system(f"rm {dump_path}")

            # 2. Create a new dump.sql of the current content of the db
            os.system(f"docker exec -t {container_id} pg_dump -U user search_engine_db > {dump_path}")

            print("SC: Successfully overwritten the old dump. Now you only need to push it to the repository!")

        except DockerException:
            print("SC: Docker is unavailable now. Please start it first and try again.")
        except Exception as error:
            print(error)

    def _get_container_id_by_image(self, image_name) -> str | None:
        """
        Returns the first ID of the container with the image `image_name`.
        When starting docker in our example we want to get the container id of "project_mse-db" to create a dump out of it
        """
        try:
            client = docker.from_env()
            containers = client.containers.list(filters={"ancestor": image_name})

            if containers:
                return containers[0].id
            else:
                return None
        except docker.errors.DockerException as e:
            print(f"Error: {e}")
            return None
