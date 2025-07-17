import sqlite3

# Connect to database
conn = sqlite3.connect("cognates_2.db")
cur = conn.cursor()

# Get a random word with its pronunciations
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
if results:
    word, lang, *_ = results[0]
    print(f"\nWord: {word}")
    print(f"Language: {lang}")
    print("\nPronunciations:")
    for _, _, trans, start, end in results:
        print(f"  {trans} ({start:.2f}-{end:.2f})")

conn.close() 