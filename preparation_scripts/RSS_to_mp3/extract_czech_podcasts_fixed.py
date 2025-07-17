import sqlite3
import json

conn = sqlite3.connect("podcastindex_feeds.db")
cursor = conn.cursor()

# First check what columns are available
cursor.execute("PRAGMA table_info(podcasts)")
columns = cursor.fetchall()
print("Available columns:")
for col in columns:
    print(f"- {col[1]}")  # Column name is at index 1

# Get all podcasts in Czech (including cs-CS variant)
cursor.execute("""
    SELECT id, title, language, description, url
    FROM podcasts 
    WHERE language IN ('cs', 'cs-CS')
""")

# Fetch all results
czech_podcasts = cursor.fetchall()

# Convert to list of dictionaries
podcasts_list = []
for podcast in czech_podcasts:
    podcasts_list.append({
        'id': podcast[0],
        'title': podcast[1],
        'language': podcast[2],
        'description': podcast[3],
        'url': podcast[4]
    })

# Write to JSON file
with open('czech_podcasts.json', 'w', encoding='utf-8') as f:
    json.dump(podcasts_list, f, ensure_ascii=False, indent=2)

print(f"\nFound {len(podcasts_list)} Czech podcasts")
print("Data has been saved to czech_podcasts.json")

conn.close() 