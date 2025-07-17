import sqlite3
import json

conn = sqlite3.connect("podcastindex_feeds.db")
cursor = conn.cursor()

# Get all podcasts in Czech
cursor.execute("""
    SELECT id, title, author, language, description, url
    FROM podcasts 
    WHERE language = 'cs'
""")

# Fetch all results
czech_podcasts = cursor.fetchall()

# Convert to list of dictionaries
podcasts_list = []
for podcast in czech_podcasts:
    podcasts_list.append({
        'id': podcast[0],
        'title': podcast[1],
        'author': podcast[2],
        'language': podcast[3],
        'description': podcast[4],
        'url': podcast[5]
    })

# Write to JSON file
with open('czech_podcasts.json', 'w', encoding='utf-8') as f:
    json.dump(podcasts_list, f, ensure_ascii=False, indent=2)

print(f"Found {len(podcasts_list)} Czech podcasts")
print("Data has been saved to czech_podcasts.json")

conn.close() 