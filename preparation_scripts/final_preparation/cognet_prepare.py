import sqlite3
import csv
from pathlib import Path
from collections import defaultdict
from concept_viz import get_concept_components
import uuid
import time

# Increase CSV field size limit
csv.field_size_limit(2147483647)  # Use maximum 32-bit integer value

# === CONFIG ===
INPUT_TSV = "CogNet-v2.0.tsv"      # path to your TSV file
DB_PATH = "cognates_2.db"         # output DB path
BATCH_SIZE = 1000  # Number of words to insert in a single batch

# === SETUP DATABASE ===
def setup_database(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Enable foreign keys and set pragmas for better performance
    cur.execute("PRAGMA foreign_keys = ON;")
    cur.execute("PRAGMA journal_mode = WAL;")  # Write-Ahead Logging for better concurrency
    cur.execute("PRAGMA synchronous = NORMAL;")  # Faster writes with reasonable safety
    cur.execute("PRAGMA cache_size = -2000;")  # Use 2MB of cache
    cur.execute("PRAGMA temp_store = MEMORY;")  # Store temp tables and indices in memory

    # Create tables
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS cognate_groups (
        component_id TEXT PRIMARY KEY,
        concept_id TEXT NOT NULL,
        description TEXT
    );

    CREATE TABLE IF NOT EXISTS words (
        word_id INTEGER PRIMARY KEY AUTOINCREMENT,
        component_id TEXT NOT NULL,
        text TEXT NOT NULL,
        language TEXT NOT NULL,
        FOREIGN KEY(component_id) REFERENCES cognate_groups(component_id)
    );

    CREATE TABLE IF NOT EXISTS pronunciations (
        pron_id INTEGER PRIMARY KEY AUTOINCREMENT,
        word_ids TEXT NOT NULL,
        transcription TEXT,
        lemma TEXT,
        wave_path TEXT,
        chopped_word_path TEXT,
        time_start REAL,
        time_end REAL,
        UNIQUE(transcription, time_start, time_end, wave_path)
    );
    """)

    conn.commit()
    return conn

def is_valid_word_pair(lang: str, word: str) -> bool:
    """Check if a language-word pair is valid for database insertion."""
    return (
        lang is not None and 
        word is not None and 
        isinstance(lang, str) and 
        isinstance(word, str) and 
        lang.strip() != "" and 
        word.strip() != ""
    )

# === INSERT COMPONENTS INTO DATABASE ===
def insert_components(conn):
    cur = conn.cursor()
    start_time = time.time()

    # Get all components from concept_viz
    print("Getting components from concept_viz...")
    components = get_concept_components()
    
    # Prepare batch inserts
    group_batch = []
    word_batch = []
    total_words = 0
    total_components = 0
    skipped_components = 0
    skipped_words = 0
    
    print("Preparing data for insertion...")
    # Process each component
    for concept_id, component in components:
        # Skip empty components
        if not component:
            skipped_components += 1
            continue
            
        # Validate all words in component
        valid_words = [(lang, word) for lang, word in component if is_valid_word_pair(lang, word)]
        if not valid_words:
            skipped_components += 1
            continue
            
        component_id = str(uuid.uuid4())
        group_batch.append((component_id, concept_id, f"Component from concept {concept_id}"))

        # Add component to cognate_groups
        cur.execute("INSERT INTO cognate_groups (component_id, concept_id, description) VALUES (?, ?, ?)", (component_id, concept_id, f"Component from concept {concept_id}"))
        
        # Add valid words to batch
        for lang, word in valid_words:
            word_batch.append((component_id, lang, word))
            total_words += 1
            
            # If batch is full, insert it
            if len(word_batch) >= BATCH_SIZE:
                try:
                    cur.executemany(
                        "INSERT INTO words (component_id, language, text) VALUES (?, ?, ?)",
                        word_batch
                    )
                except sqlite3.IntegrityError as e:
                    print(f"Error inserting word batch: {e}")
                    print(f"First problematic word: {word_batch[0]}")
                    raise
                word_batch = []
        
        total_components += 1
        if total_components % 100 == 0:
            print(f"Processed {total_components} components, {total_words} words...")
            print(f"Skipped {skipped_components} invalid components, {skipped_words} invalid words")
    
    print("Inserting remaining data...")
    # Insert remaining groups
    try:
        cur.executemany(
            "INSERT OR IGNORE INTO cognate_groups (component_id, concept_id, description) VALUES (?, ?, ?)",
            group_batch
        )
    except sqlite3.IntegrityError as e:
        print(f"Error inserting groups: {e}")
        print(f"First problematic group: {group_batch[0]}")
        raise
    
    # Insert remaining words
    if word_batch:
        try:
            cur.executemany(
                "INSERT INTO words (component_id, language, text) VALUES (?, ?, ?)",
                word_batch
            )
        except sqlite3.IntegrityError as e:
            print(f"Error inserting final word batch: {e}")
            print(f"First problematic word: {word_batch[0]}")
            raise
    
    conn.commit()
    end_time = time.time()
    print(f"\nFinal Statistics:")
    print(f"✅ Inserted {total_components} components with {total_words} words in {end_time - start_time:.2f} seconds")
    print(f"❌ Skipped {skipped_components} invalid components")
    print(f"❌ Skipped {skipped_words} invalid words")

# === MAIN ===
def main():
    print("🔧 Setting up database...")
    conn = setup_database(DB_PATH)

    print("📥 Inserting cognate groups and words from components...")
    insert_components(conn)

    print("✅ Done! Database saved to", DB_PATH)
    conn.close()

if __name__ == "__main__":
    main()
