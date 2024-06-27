import psycopg2
from typing import List

from db.DocumentEntry import DocumentEntry


class DocumentRepository:

    def __init__(self):
        try:
            # Connect to your PostgreSQL database. Here are the credentials written so the machine knows where connect to.
            self.connection = psycopg2.connect(
                user="user",
                password="user_pw",
                host="localhost",
                port="5432",
                database="search_engine_db"
            )

            # Create a cursor object. We need this to execute queries.
            self.cursor = self.connection.cursor()

            print("SC: Connected to the db. Now you can go and build the best search engine around!")  # for sanity check (SC)
        except (Exception, psycopg2.Error) as error:
            print("SC: Connecting to PostgreSQL did not work. Fix it", error)

    def readAllDocuments(self) -> list[DocumentEntry]:
        self.cursor.execute("SELECT * FROM documents")
        query_result = self.cursor.fetchall()
        return query_result
