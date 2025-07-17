import json
import sqlite3
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_language_codes():
    with open('language_codes.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def add_pronunciation(cur, word_ids, word_info, json_data):
    """Add a pronunciation for multiple word_ids."""
    # Convert word_ids list to comma-separated string
    word_ids_str = ','.join(map(str, word_ids))
    
    try:
        cur.execute("""
            INSERT INTO pronunciations 
            (word_ids, transcription, lemma, wave_path, time_start, time_end)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            word_ids_str,
            word_info['word'].strip(),
            word_info.get('lemma', '').strip() or word_info['word'].strip(),
            json_data['metadata']['source_file'],  # Get wave_path from metadata
            word_info['start'],
            word_info['end']
        ))
        logger.info(f"Successfully added pronunciation for word_ids: {word_ids_str}")
        return True
    except sqlite3.IntegrityError as e:
        logger.error(f"Error adding pronunciation: {e}")
        return False

def debug_word_matching(json_data=None):
    # Load language codes
    language_codes = load_language_codes()
    
    # Connect to database
    conn = sqlite3.connect('cognates_2.db')
    cur = conn.cursor()
    
    # If no json_data provided, load the test file
    if json_data is None:
        json_path = Path("../WhisperVault/transcriptions_lemmatized/65729845_en_7309884_feed_lemmatized.json")
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    
    # Get language code
    two_letter_code = json_data['metadata']['language']
    three_letter_code = language_codes.get(two_letter_code, two_letter_code)
    logger.info(f"Language code conversion: {two_letter_code} -> {three_letter_code}")
    
    # Debug words until first match
    for i, word_info in enumerate(json_data['words']):
        word = word_info['word'].strip()
        lemma = word_info.get('lemma', '').strip()
        if not lemma or lemma == '_':
            lemma = word
        
        logger.info(f"\nWord {i+1}:")
        logger.info(f"Original word: '{word}'")
        logger.info(f"Lemma: '{lemma}'")
        
        # Try to find matches in database
        cur.execute("""
            SELECT word_id, language, text 
            FROM words 
            WHERE language = ? AND (text = ? OR text = ?)
        """, (three_letter_code, word, lemma))
        
        matches = cur.fetchall()
        if matches:
            logger.info("Found matches:")
            word_ids = []
            for match in matches:
                logger.info(f"  ID: {match[0]}, Lang: {match[1]}, Text: '{match[2]}'")
                word_ids.append(match[0])
            
            # Add pronunciation for all matching word_ids
            if add_pronunciation(cur, word_ids, word_info, json_data):
                logger.info("Successfully added pronunciation and terminating.")
                conn.commit()
                break
        else:
            logger.info("No matches found")
            
            # Let's check what's in the database for this language
            cur.execute("""
                SELECT DISTINCT text 
                FROM words 
                WHERE language = ? 
                LIMIT 5
            """, (three_letter_code,))
            sample_words = cur.fetchall()
            logger.info(f"Sample words in database for {three_letter_code}:")
            for sample in sample_words:
                logger.info(f"  '{sample[0]}'")
    
    conn.close()

if __name__ == "__main__":
    debug_word_matching() 