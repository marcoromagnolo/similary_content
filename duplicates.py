import mysql.connector
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from settings import DB_SETTINGS
from scipy.sparse import csr_matrix, vstack


class Similarity:
    def __init__(self, target_website_id):
        self.target_website_id = target_website_id
        self.vectorizer = None
        self.tfidf_matrix = None


    def get_pages(self, db_cursor):
        db_cursor = self.db_connection.cursor()
        db_cursor.execute("SELECT id, content FROM target_page WHERE similarity IS NULL")
        pages = db_cursor.fetchall()
        db_cursor.close()
        return pages

    def get_document(self, page_id):
        """Get a document."""
        db_cursor = self.db_connection.cursor()
        db_cursor.execute(
            "SELECT GROUP_CONCAT(tk.value SEPARATOR ' ') AS keywords "
            "FROM target_keyword tk LEFT JOIN target_page tp ON tp.id = tk.target_page_id "
            "WHERE tk.target_page_id = %s AND tp.similarity is NULL",
            (page_id,)
        )
        result = db_cursor.fetchone()
        db_cursor.close()
        return result[0] if result else ""


    def get_all_keywords(self):
        """Get all keywords for the target website grouped by page."""
        db_cursor = self.db_connection.cursor()
        db_cursor.execute(
            "SELECT GROUP_CONCAT(tk.value SEPARATOR ' ') AS keywords "
            "FROM target_keyword tk "
            "LEFT JOIN target_page tp ON tp.id = tk.target_page_id "
            "WHERE tp.target_website_id = %s AND tp.similarity IS NOT NULL "
            "GROUP BY tk.target_page_id",
            (self.target_website_id,)
        )
        keywords = [row[0] for row in db_cursor.fetchall()]
        db_cursor.close()
        return keywords


    def load_vocabulary(self):
        """Load vocabulary from the database."""
        db_cursor = self.db_connection.cursor()
        db_cursor.execute("SELECT term, index_in_vector FROM vocabulary")
        vocabulary = {term: index for term, index in db_cursor.fetchall()}
        print("Loaded Vocabulary:", vocabulary)
        db_cursor.close()
        return vocabulary


    def update_vocabulary(self, documents):
        """Update vocabulary"""
        
        db_cursor = self.db_connection.cursor()
        db_cursor.execute("SELECT MAX(index_in_vector) FROM vocabulary LIMIT 1")
        v = db_cursor.fetchone()
        max_index = v[0]
        print("Loaded Max index:", max_index)

        new_vectorizer = TfidfVectorizer()
        new_vectorizer.fit(documents)
        
        new_index = -1 if max_index is None else max_index
        for term, index in new_vectorizer.vocabulary_.items():
            new_index += 1
            db_cursor.execute(
                "INSERT IGNORE INTO vocabulary (term, index_in_vector) VALUES (%s, %s)",
                (term, new_index)
            )
        self.db_connection.commit()    

        db_cursor.close()      


    def load_matrix(self, vocabulary):
        """Load TF-IDF matrnew_tfidf_matrixix and vocabulary for the target website."""
        db_cursor = self.db_connection.cursor()

        # Load vocabulary
        self.vectorizer = TfidfVectorizer(vocabulary=vocabulary) 

        # Load TF-IDF matrix
        db_cursor.execute(
            "SELECT row_index, col_index, value FROM tfidf_matrix WHERE target_website_id = %s",
            (self.target_website_id,)
        )
        data = db_cursor.fetchall()
        if data:
            # Reconstruct the sparse matrix
            rows, cols, values = zip(*data)
            num_rows = max(rows) + 1
            num_cols = max(cols) + 1
            self.tfidf_matrix = csr_matrix((values, (rows, cols)), shape=(num_rows, num_cols))
            print("Loaded TF-IDF matrix:", self.tfidf_matrix.toarray())
        else:
            print("Loaded TF-IDF matrix is empty.")
        db_cursor.close()
        

    def update_matrix(self, new_tfidf_matrix):
        """Save new rows to TF-IDF matrix"""

        db_cursor = self.db_connection.cursor()
        rows, cols = new_tfidf_matrix.nonzero()
        for row, col in zip(rows, cols):
            value = new_tfidf_matrix[row, col]
            db_cursor.execute(
                """
                INSERT INTO tfidf_matrix (target_website_id, row_index, col_index, value) 
                VALUES (%s, %s, %s, %s) 
                ON DUPLICATE KEY UPDATE value = %s
                """,
                (self.target_website_id, row.item(), col.item(), value.item(), value.item())
            )
            
        # Append new rows to the in-memory matrix
        if self.tfidf_matrix is None:
            self.tfidf_matrix = new_tfidf_matrix
        else:
            self.tfidf_matrix = vstack([self.tfidf_matrix, new_tfidf_matrix])

        print("Incremental update: TF-IDF matrix and vocabulary updated.")
        self.db_connection.commit()
        db_cursor.close()        


    def update_page_similarity(self, db_cursor, page_id, document):
        if document:
            documents = [document]

            new_tfidf_matrix = self.vectorizer.transform(documents)

            if self.tfidf_matrix is None:
                similarity_score = 0.0
            else:
                similarity = cosine_similarity(new_tfidf_matrix, self.tfidf_matrix)
                similarity_score = similarity.max().item()
            
            # Update the similarity in the database
            db_cursor.execute(
                "UPDATE target_page SET similarity = %s WHERE id = %s AND similarity IS NULL",
                (similarity_score, page_id)
            )

            self.update_matrix(new_tfidf_matrix)

            self.db_connection.commit()


    def update_target_page(self, page_id):
        """Update similarity for a specific target page."""

        self.db_connection = mysql.connector.connect(**DB_SETTINGS)
        db_cursor = self.db_connection.cursor()

        # Compute similarity
        document = self.get_document(page_id)
        if document:
            self.update_vocabulary([document])
            vocabulary = self.load_vocabulary()
            self.load_matrix(vocabulary)

            self.update_page_similarity(db_cursor, page_id, document)

        db_cursor.close()
        self.db_connection.close()


    def update_target_pages(self):
        """Update similarity for all target pages."""

        self.db_connection = mysql.connector.connect(**DB_SETTINGS)
        db_cursor = self.db_connection.cursor()
        
        page_ids = []
        documents = []
        for (page_id, document) in self.get_pages(db_cursor):
            page_ids.append(page_id)
            documents.append(document)
        db_cursor.close()

        self.update_vocabulary(documents)
        vocabulary = self.load_vocabulary
        self.load_matrix(vocabulary)

        for i in range(len(documents)):
            # Compute similarity
            self.update_page_similarity(db_cursor, page_ids[i], documents[i])

        db_cursor.close()
        self.db_connection.close()
