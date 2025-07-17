import sqlite3
import time
from datetime import datetime

def print_progress(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def analyze_tables():
    print_progress("Connecting to database...")
    conn = sqlite3.connect("cognates_2.db")
    conn.execute("PRAGMA busy_timeout = 5000")  # 5 second timeout
    cur = conn.cursor()
    
    # Get table sizes
    print("\nTable Sizes:")
    print("-" * 50)
    print_progress("Getting list of tables...")
    cur.execute("""
        SELECT name
        FROM sqlite_master 
        WHERE type='table'
    """)
    tables = cur.fetchall()
    
    for table in tables:
        table_name = table[0]
        print_progress(f"Counting rows in {table_name}...")
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cur.fetchone()[0]
            print(f"{table_name}: {count:,} rows")
        except sqlite3.Error as e:
            print(f"Error counting {table_name}: {e}")
            continue
        
        # Get sample of data
        if table_name == 'pronunciations':
            print("\nSample from pronunciations table:")
            try:
                cur.execute("""
                    SELECT pron_id, word_ids, transcription 
                    FROM pronunciations 
                    LIMIT 3
                """)
                samples = cur.fetchall()
                for sample in samples:
                    print(f"  ID: {sample[0]}")
                    print(f"  Word IDs: {sample[1]}")
                    print(f"  Transcription: {sample[2]}")
                    print()
            except sqlite3.Error as e:
                print(f"Error getting samples: {e}")
        
        # Check if there are indexes
        print_progress(f"Checking indexes on {table_name}...")
        try:
            cur.execute(f"PRAGMA index_list({table_name})")
            indexes = cur.fetchall()
            if indexes:
                print(f"Indexes on {table_name}:")
                for idx in indexes:
                    print(f"  - {idx[1]}")
        except sqlite3.Error as e:
            print(f"Error checking indexes: {e}")
        print()

    # Analyze the slow query
    print("\nAnalyzing the slow query:")
    print("-" * 50)
    
    # First, let's see how many words have pronunciations
    print_progress("Counting words with pronunciations...")
    try:
        cur.execute("""
            SELECT COUNT(DISTINCT w.word_id)
            FROM words w
            JOIN pronunciations p ON p.word_ids LIKE '%' || w.word_id || '%'
        """)
        words_with_pron = cur.fetchone()[0]
        print(f"Words with pronunciations: {words_with_pron:,}")
    except sqlite3.Error as e:
        print(f"Error counting words: {e}")
    
    # Check how many pronunciations per word on average
    print_progress("Calculating average pronunciations per word...")
    try:
        cur.execute("""
            SELECT AVG(pron_count)
            FROM (
                SELECT w.word_id, COUNT(*) as pron_count
                FROM words w
                JOIN pronunciations p ON p.word_ids LIKE '%' || w.word_id || '%'
                GROUP BY w.word_id
            )
        """)
        avg_prons = cur.fetchone()[0]
        print(f"Average pronunciations per word: {avg_prons:.2f}")
    except sqlite3.Error as e:
        print(f"Error calculating average: {e}")
    
    # Time the original query
    print("\nTiming the original query...")
    print_progress("Running test query...")
    start = time.time()
    try:
        cur.execute("""
            WITH random_word AS (
                SELECT w.word_id, w.text, w.language
                FROM words w
                JOIN pronunciations p ON p.word_ids LIKE '%' || w.word_id || '%'
                ORDER BY RANDOM()
                LIMIT 1
            )
            SELECT 
                rw.text,
                rw.language,
                p.transcription,
                p.time_start,
                p.time_end
            FROM random_word rw
            JOIN pronunciations p ON p.word_ids LIKE '%' || rw.word_id || '%'
            ORDER BY p.time_start
        """)
        results = cur.fetchall()
        end = time.time()
        print(f"Query took {end - start:.2f} seconds")
        print(f"Found {len(results)} pronunciations")
    except sqlite3.Error as e:
        print(f"Error running test query: {e}")
    
    conn.close()
    print_progress("Analysis complete!")

if __name__ == "__main__":
    try:
        analyze_tables()
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nError during analysis: {e}") 