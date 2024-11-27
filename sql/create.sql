create table if not exists vocabulary (
	term VARCHAR(255), 
	index_in_vector INT,
	primary key(term)
);

create table if not exists tfidf_matrix (
    row_index INT,
    col_index INT,
    value FLOAT,
	target_website_id INT,
    PRIMARY KEY (row_index, col_index, target_website_id)
);

CREATE TABLE serialized_objects (
    id INT AUTO_INCREMENT PRIMARY KEY,
    target_website_id INT NOT NULL,
    object_name VARCHAR(255) NOT NULL,
    data LONGBLOB NOT NULL,
    UNIQUE KEY (target_website_id, object_name)
); 
