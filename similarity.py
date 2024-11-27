import mysql.connector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import io
from settings import DB_SETTINGS


class Similarity:
    def __init__(self, target_website_id):
        self.target_website_id = target_website_id
        self.vectorizer = self.load_object("vectorizer")
        self.tfidf_matrix = self.load_object("tfidf_matrix")
        self.page_ids = self.load_object("page_ids")
        if not self.vectorizer:
            self.db_connection = mysql.connector.connect(**DB_SETTINGS)
            self.build_vectorizer()
            self.db_connection.close()
        

    def serialize_object(self, obj):
        """Serialize an object to binary using joblib."""
        buffer = io.BytesIO()
        joblib.dump(obj, buffer)
        buffer.seek(0)
        return buffer.read()
    

    def deserialize_object(self, binary_data):
        """Deserialize an object from binary using joblib."""
        buffer = io.BytesIO(binary_data)
        return joblib.load(buffer)
    

    def save_object(self, object_name, obj):
        """Save a serialized object to the database."""
        binary_data = self.serialize_object(obj)
        db_cursor = self.db_connection.cursor()
        db_cursor.execute(
            """
            INSERT INTO serialized_objects (target_website_id, object_name, data) 
            VALUES (%s, %s, %s) 
            ON DUPLICATE KEY UPDATE data = %s
            """,
            (self.target_website_id, object_name, binary_data, binary_data)
        )
        self.db_connection.commit()
        db_cursor.close()


    def load_object(self, object_name):
        """Load a serialized object from the database."""
        self.db_connection = mysql.connector.connect(**DB_SETTINGS)
        db_cursor = self.db_connection.cursor()
        db_cursor.execute(
            "SELECT data FROM serialized_objects WHERE target_website_id = %s AND object_name = %s",
            (self.target_website_id, object_name)
        )
        result = db_cursor.fetchone()
        db_cursor.close()
        self.db_connection.close()
        if result:
            return self.deserialize_object(result[0])
        else:
            return None
        

    def build_vectorizer(self):
        """Retrieve a document for the given page ID."""
        db_cursor = self.db_connection.cursor()
        db_cursor.execute(
            "SELECT id, CONCAT(tp.title, ' ', tp.content) AS document "
            "FROM target_page tp WHERE tp.similarity IS NOT NULL"
        )

        rows = db_cursor.fetchall()
        documents = [row[1] for row in rows]
        self.page_ids = [row[0] for row in rows]

        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.save_object("vectorizer", self.vectorizer)
        self.save_object("tfidf_matrix", self.tfidf_matrix)
        self.save_object("page_ids", self.page_ids)
        db_cursor.close()


    def get_document(self, page_id):
        """Retrieve a document for the given page ID."""
        db_cursor = self.db_connection.cursor()
        db_cursor.execute(
            "SELECT CONCAT(tp.title, ' ', tp.content) AS document "
            "FROM target_page tp "
            "WHERE tp.id = %s AND tp.similarity IS NULL",
            (page_id,)
        )
        result = db_cursor.fetchone()
        db_cursor.close()
        return result[0] if result else ""


    def update_page_similarity(self, page_id):
        """Update the similarity for a specific page."""
        self.db_connection = mysql.connector.connect(**DB_SETTINGS)
        db_cursor = self.db_connection.cursor()

        # Retrieve the document for the page
        document = self.get_document(page_id)
        if not document:
            print(f"No document found for page ID {page_id}.")
            return

        new_tfidf_matrix = self.vectorizer.transform([document])

        # Compute cosine similarity with all existing documents
        similarity = cosine_similarity(new_tfidf_matrix, self.tfidf_matrix)

        # Get the max similarity score and its index
        max_similarity_score = similarity.max().item()
        max_similarity_index = similarity.argmax()

        # Find the corresponding page ID using the index
        most_similar_page_id = self.page_ids[max_similarity_index]

        print(f"Most similar page ID for page id:{page_id} is the page with id:{most_similar_page_id} with similarity {max_similarity_score}")
     

        # Update similarity in the database
        db_cursor.execute(
            "UPDATE target_page SET similarity = %s WHERE id = %s AND similarity IS NULL",
            (max_similarity_score, page_id)
        )

        # Save the updated vectorizer
        self.save_object("vectorizer", self.vectorizer)

        # Save updates to the database
        self.save_object("tfidf_matrix", self.tfidf_matrix)

        self.db_connection.commit()
        db_cursor.close()
        self.db_connection.close()

    def update_pages_similarity(self, page_ids):
        """Update similarity scores for multiple pages."""
        self.db_connection = mysql.connector.connect(**DB_SETTINGS)
        db_cursor = self.db_connection.cursor()

        for page_id in page_ids:
            document = self.get_document(page_id)
            if not document:
                print(f"No document found for page ID {page_id}.")
                continue

            new_tfidf_matrix = self.vectorizer.transform([document])
            similarity = cosine_similarity(new_tfidf_matrix, self.tfidf_matrix)
            similarity_score = similarity.max().item()

            db_cursor.execute(
                "UPDATE target_page SET similarity = %s WHERE id = %s AND similarity IS NULL",
                (similarity_score, page_id)
            )

        self.db_connection.commit()
        db_cursor.close()
        self.db_connection.close()    