import sqlite3
import json
from tqdm import tqdm

conn = sqlite3.connect("podcastindex_feeds.db")
cursor = conn.cursor()

# Get total rows
cursor.execute("SELECT COUNT(*) FROM podcasts")
total_rows = cursor.fetchone()[0]

# Initialize list to store all languages
all_languages = []

batch_size = 1000
for offset in tqdm(range(0, total_rows, batch_size)):
    cursor.execute(f"SELECT language FROM podcasts LIMIT {batch_size} OFFSET {offset}")
    rows = cursor.fetchall()
    # Add languages to our list
    all_languages.extend([row[0] for row in rows])

# Write to JSON file
with open('languages.json', 'w', encoding='utf-8') as f:
    json.dump(all_languages, f, ensure_ascii=False, indent=2)

conn.close()
