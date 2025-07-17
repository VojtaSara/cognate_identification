import json
import sqlite3
from pathlib import Path
import logging
import time
from tqdm import tqdm
from typing import List, Dict, Generator
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_language_codes() -> Dict[str, str]:
    """Load language code mappings from JSON file."""
    with open('language_codes.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def get_json_files(json_dir: str) -> List[Path]:
    """Get all lemmatized JSON files from the directory."""
    json_path = Path(json_dir)
    return list(json_path.glob('*_lemmatized.json'))

def load_json_file(file_path: Path) -> Dict:
    """Load and parse a single JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(file_path: Path, data: Dict):
    """Save JSON data to file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def add_pronunciation(cur, word_ids: List[int], word_info: Dict, json_data: Dict) -> bool:
    """Add a pronunciation for multiple word_ids."""
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
            json_data['metadata']['source_file'],
            word_info['start'],
            word_info['end']
        ))
        return True
    except sqlite3.IntegrityError:
        return False

def process_json_files(json_dir: str, db_path: str):
    """Process all JSON files and add pronunciations to database."""
    # Load language codes
    language_codes = load_language_codes()
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Get all JSON files
    json_files = get_json_files(json_dir)
    logger.info(f"\n{'='*80}\nFound {len(json_files)} JSON files to process\n{'='*80}")
    
    # Statistics
    total_words_processed = 0
    total_words_matched = 0
    total_pronunciations_added = 0
    start_time = time.time()
    
    # Process each JSON file
    for file_path in tqdm(json_files, desc="Processing JSON files", position=0):
        try:
            # Load JSON data
            json_data = load_json_file(file_path)

            # Check if already processed
            if json_data.get('processed_for_db', False):
                logger.info(f"Skipping {file_path.name}: already processed.")
                continue

            two_letter_code = json_data['metadata']['language']
            three_letter_code = language_codes.get(two_letter_code, two_letter_code)
            
            # Process words in this file
            file_words_processed = 0
            file_words_matched = 0
            file_pronunciations_added = 0
            
            # Create progress bar for words in this file
            words_pbar = tqdm(json_data['words'], 
                            desc=f"Processing {file_path.name}", 
                            position=1, 
                            leave=False)
            
            for word_info in words_pbar:
                file_words_processed += 1
                total_words_processed += 1
                
                word = word_info['word'].strip()
                lemma = word_info.get('lemma', '').strip()
                if not lemma or lemma == '_':
                    lemma = word
                
                # Find matches in database
                cur.execute("""
                    SELECT word_id, language, text 
                    FROM words 
                    WHERE language = ? AND (text = ? OR text = ?)
                """, (three_letter_code, word, lemma))
                
                matches = cur.fetchall()
                if matches:
                    file_words_matched += 1
                    total_words_matched += 1
                    
                    # Collect word_ids
                    word_ids = [match[0] for match in matches]
                    
                    # Add pronunciation
                    if add_pronunciation(cur, word_ids, word_info, json_data):
                        file_pronunciations_added += 1
                        total_pronunciations_added += 1
                
                # Update progress bar description
                words_pbar.set_description(
                    f"File: {file_path.name} | "
                    f"Words: {file_words_processed} | "
                    f"Matches: {file_words_matched} | "
                    f"Added: {file_pronunciations_added}"
                )
            
            words_pbar.close()
            
            # Mark file as processed and save
            json_data['processed_for_db'] = True
            save_json_file(file_path, json_data)
            
            # Commit after each file
            conn.commit()
            
            # Log file statistics
            logger.info(f"\n{'-'*80}")
            logger.info(f"File: {file_path.name}")
            logger.info(f"Words processed: {file_words_processed}")
            logger.info(f"Words matched: {file_words_matched}")
            logger.info(f"Pronunciations added: {file_pronunciations_added}")
            logger.info(f"{'-'*80}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
    
    # Calculate and log final statistics
    elapsed_time = time.time() - start_time
    logger.info(f"\n{'='*80}")
    logger.info("FINAL STATISTICS:")
    logger.info(f"Total words processed: {total_words_processed}")
    logger.info(f"Total words matched: {total_words_matched}")
    logger.info(f"Total pronunciations added: {total_pronunciations_added}")
    logger.info(f"Processing speed: {total_words_processed/elapsed_time:.2f} words/second")
    logger.info(f"Insertion speed: {total_pronunciations_added/elapsed_time:.2f} pronunciations/second")
    logger.info(f"Total time: {elapsed_time:.2f} seconds")
    logger.info(f"{'='*80}")
    
    conn.close()

if __name__ == "__main__":
    # Configuration
    JSON_DIR = "../WhisperVault/transcriptions"
    DB_PATH = "cognates_3.db"
    
    # Process all JSON files
    process_json_files(JSON_DIR, DB_PATH) 