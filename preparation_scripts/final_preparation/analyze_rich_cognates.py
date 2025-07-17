import sqlite3
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import json
from tqdm import tqdm

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

def get_rich_cognate_groups(cur, min_pronunciations: int = 5, min_languages: int = 3) -> List[Tuple[str, int, int]]:
    """
    Find cognate groups with many pronunciations across multiple languages.
    Returns list of (component_id, pronunciation_count, language_count) tuples.
    """
    cur.execute("""
        WITH pronunciation_counts AS (
            SELECT 
                c.component_id,
                COUNT(DISTINCT p.pron_id) as pron_count,
                COUNT(DISTINCT w.language) as lang_count
            FROM cognate_groups c
            JOIN words w ON c.component_id = w.component_id
            JOIN pronunciations p ON p.word_ids LIKE '%' || w.word_id || '%'
            GROUP BY c.component_id
            HAVING pron_count >= ? AND lang_count >= ?
        )
        SELECT 
            pc.component_id,
            pc.pron_count,
            pc.lang_count,
            c.description
        FROM pronunciation_counts pc
        JOIN cognate_groups c ON pc.component_id = c.component_id
        ORDER BY pc.pron_count DESC, pc.lang_count DESC
        LIMIT 20
    """, (min_pronunciations, min_languages))
    
    return cur.fetchall()

def get_cognate_group_details(cur, component_id: str) -> Dict:
    """Get detailed information about a specific cognate group."""
    # Get basic group info
    cur.execute("""
        SELECT description, concept_id
        FROM cognate_groups
        WHERE component_id = ?
    """, (component_id,))
    group_info = cur.fetchone()
    
    # Get words and their pronunciations
    cur.execute("""
        SELECT 
            w.word_id,
            w.language,
            w.text,
            p.pron_id,
            p.transcription,
            p.lemma,
            p.wave_path,
            p.time_start,
            p.time_end
        FROM words w
        LEFT JOIN pronunciations p ON p.word_ids LIKE '%' || w.word_id || '%'
        WHERE w.component_id = ?
        ORDER BY w.language, w.text
    """, (component_id,))
    
    words = cur.fetchall()
    
    # Organize data
    result = {
        'component_id': component_id,
        'description': group_info[0],
        'concept_id': group_info[1],
        'words': defaultdict(list)
    }
    
    for word in words:
        word_info = {
            'word_id': word[0],
            'language': word[1],
            'text': word[2]
        }
        
        if word[3]:  # If there's a pronunciation
            pron_info = {
                'pron_id': word[3],
                'transcription': word[4],
                'lemma': word[5],
                'wave_path': word[6],
                'time_start': word[7],
                'time_end': word[8]
            }
            word_info['pronunciation'] = pron_info
        
        result['words'][word[1]].append(word_info)
    
    return result

def analyze_rich_cognates(db_path: str, output_dir: str = "rich_cognates"):
    """Analyze and save information about rich cognate groups."""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Load language codes for display
    language_codes = load_language_codes()
    reverse_codes = {v: k for k, v in language_codes.items()}
    
    try:
        # Find rich cognate groups
        logger.info("Finding rich cognate groups...")
        rich_groups = get_rich_cognate_groups(cur)
        
        if not rich_groups:
            logger.info("No rich cognate groups found with current criteria.")
            return
        
        # Process each group
        for group in tqdm(rich_groups, desc="Processing cognate groups"):
            component_id, pron_count, lang_count, description = group
            
            # Get detailed information
            details = get_cognate_group_details(cur, component_id)
            
            # Create a readable summary
            summary = f"""Cognate Group Analysis
====================
Component ID: {component_id}
Description: {description}
Concept ID: {details['concept_id']}
Total Pronunciations: {pron_count}
Languages: {lang_count}

Words and Pronunciations by Language:
"""
            
            for lang, words in details['words'].items():
                lang_display = reverse_codes.get(lang, lang)
                summary += f"\n{lang_display} ({lang}):\n"
                for word in words:
                    summary += f"  - {word['text']}"
                    if 'pronunciation' in word:
                        pron = word['pronunciation']
                        summary += f"\n    Pronunciation: {pron['transcription']}"
                        summary += f"\n    Time: {pron['time_start']:.2f}-{pron['time_end']:.2f}"
                        summary += f"\n    File: {pron['wave_path']}"
                    summary += "\n"
            
            # Save to file
            output_file = output_path / f"{component_id}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            # Also save raw JSON for potential further processing
            json_file = output_path / f"{component_id}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(details, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nAnalysis complete. Results saved in {output_dir}/")
        logger.info(f"Found {len(rich_groups)} rich cognate groups.")
        
    finally:
        conn.close()

if __name__ == "__main__":
    DB_PATH = "cognates_2.db"
    analyze_rich_cognates(DB_PATH) 