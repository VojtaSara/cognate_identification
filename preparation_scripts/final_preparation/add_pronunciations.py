# Create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS pronunciations (
    pronunciation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    transcription TEXT NOT NULL,
    lemmatization TEXT,
    wave_file TEXT,
    time_start REAL,
    time_end REAL
);

CREATE TABLE IF NOT EXISTS pronunciation_words (
    pronunciation_id INTEGER NOT NULL,
    word_id INTEGER NOT NULL,
    PRIMARY KEY (pronunciation_id, word_id),
    FOREIGN KEY (pronunciation_id) REFERENCES pronunciations(pronunciation_id),
    FOREIGN KEY (word_id) REFERENCES words(word_id)
);
""") 