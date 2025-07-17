import sqlite3
import json
from pathlib import Path
from typing import Dict, Optional
import time
import random

def load_language_codes() -> Dict[str, str]:
    """Load language code mappings from JSON file."""
    with open('language_codes.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def get_cognate_group(db_path: str, component_id: str) -> Optional[Dict]:
    """Get detailed information about a specific cognate group."""
    start_time = time.time()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    try:
        # Get basic group info
        print("Fetching basic group info...")
        group_start = time.time()
        cur.execute("""
            SELECT description, concept_id
            FROM cognate_groups
            WHERE component_id = ?
        """, (component_id,))
        group_info = cur.fetchone()
        print(f"Group info fetch took: {time.time() - group_start:.2f} seconds")
        
        if not group_info:
            print(f"No cognate group found with component_id: {component_id}")
            return None
        
        # Get words and their pronunciations
        print("\nFetching words and pronunciations...")
        words_start = time.time()
        
        # First get all words for this component
        cur.execute("""
            SELECT word_id, language, text
            FROM words
            WHERE component_id = ?
        """, (component_id,))
        words = cur.fetchall()
        print(f"Words fetch took: {time.time() - words_start:.2f} seconds")
        
        # Then get pronunciations for these words
        print("\nFetching pronunciations...")
        pron_start = time.time()
        
        # Get all pronunciations and filter in Python
        cur.execute("""
            SELECT pron_id, word_ids, transcription, lemma, wave_path, time_start, time_end
            FROM pronunciations
        """)
        pronunciations = cur.fetchall()
        
        # Create pronunciation lookup with exact word_id matching
        pron_lookup = {}
        word_ids = {w[0] for w in words}  # Set of word_ids for faster lookup
        
        for pron in pronunciations:
            pron_word_ids = set(pron[1].split(','))  # Split word_ids into set
            matching_word_ids = pron_word_ids.intersection(word_ids)  # Find matching word_ids
            
            for word_id in matching_word_ids:
                if word_id not in pron_lookup:
                    pron_lookup[word_id] = []
                pron_lookup[word_id].append({
                    'pron_id': pron[0],
                    'transcription': pron[2],
                    'lemma': pron[3],
                    'wave_path': pron[4],
                    'time_start': pron[5],
                    'time_end': pron[6]
                })
        
        print(f"Pronunciations fetch took: {time.time() - pron_start:.2f} seconds")
        
        # Load language codes for display
        language_codes = load_language_codes()
        reverse_codes = {v: k for k, v in language_codes.items()}
        
        # Organize data
        result = {
            'component_id': component_id,
            'description': group_info[0],
            'concept_id': group_info[1],
            'words': {}
        }
        
        # Print summary
        print(f"\nCognate Group: {component_id}")
        print(f"Description: {group_info[0]}")
        print(f"Concept ID: {group_info[1]}")
        print("\nWords and Pronunciations:")
        print("=" * 50)
        
        current_lang = None
        for word in words:
            word_id, lang, text = word
            if lang != current_lang:
                current_lang = lang
                lang_display = reverse_codes.get(lang, lang)
                print(f"\n{lang_display} ({lang}):")
            
            print(f"  - {text}")
            if word_id in pron_lookup:
                for pron in pron_lookup[word_id]:
                    print(f"    Pronunciation: {pron['transcription']}")
                    print(f"    Time: {pron['time_start']:.2f}-{pron['time_end']:.2f}")
                    print(f"    File: {pron['wave_path']}")
            
            # Store in result dict
            if lang not in result['words']:
                result['words'][lang] = []
            
            word_info = {
                'word_id': word_id,
                'text': text
            }
            
            if word_id in pron_lookup:
                word_info['pronunciations'] = pron_lookup[word_id]
            
            result['words'][lang].append(word_info)
        
        total_time = time.time() - start_time
        print(f"\nTotal execution time: {total_time:.2f} seconds")
        return result
        
    finally:
        conn.close()

def get_word_pronunciations(db_path: str):
    """Get a random word and all its pronunciations using a single SQL query."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Get a random word with its pronunciations in a single query
    cur.execute("""
        WITH random_word AS (
            SELECT w.word_id, w.text, w.language, c.description
            FROM words w
            JOIN cognate_groups c ON w.component_id = c.component_id
            JOIN pronunciations p ON p.word_ids LIKE '%' || w.word_id || '%'
            ORDER BY RANDOM()
            LIMIT 1
        )
        SELECT 
            rw.text,
            rw.language,
            rw.description,
            p.transcription,
            p.time_start,
            p.time_end,
            p.wave_path
        FROM random_word rw
        JOIN pronunciations p ON p.word_ids LIKE '%' || rw.word_id || '%'
        ORDER BY p.time_start
    """)
    
    results = cur.fetchall()
    if not results:
        print("No words with pronunciations found.")
        return
    
    # Print results
    word, lang, group, *_ = results[0]
    print(f"\nWord: {word}")
    print(f"Language: {lang}")
    print(f"Cognate Group: {group}")
    print("\nPronunciations:")
    print("-" * 50)
    
    for _, _, _, trans, start, end, file in results:
        print(f"\n  {trans}")
        print(f"  Time: {start:.2f}-{end:.2f}")
        print(f"  File: {file}")
    
    conn.close()

def get_random_word_id(db_path: str) -> str:
    """Get a random word ID that has pronunciations."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    try:
        cur.execute("""
            SELECT w.word_id
            FROM words w
            JOIN pronunciations p ON p.word_ids LIKE '%' || w.word_id || '%'
            ORDER BY RANDOM()
            LIMIT 1
        """)
        result = cur.fetchone()
        return result[0] if result else None
    finally:
        conn.close()

if __name__ == "__main__":
    DB_PATH = "cognates_2.db"
    
    # Get a random word with pronunciations
    word_id = get_random_word_id(DB_PATH)
    if word_id:
        print(f"Found word with ID: {word_id}")
        get_word_pronunciations(DB_PATH)
    else:
        print("No words with pronunciations found in the database.") 