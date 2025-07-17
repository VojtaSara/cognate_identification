import sqlite3
import json
import aiohttp
import asyncio
import os
from datetime import datetime, timedelta
from urllib.parse import urlparse
import time

# Create directory for RSS backups if it doesn't exist
RSS_BACKUP_DIR = "rss_backups"
if not os.path.exists(RSS_BACKUP_DIR):
    os.makedirs(RSS_BACKUP_DIR)

async def download_rss(session, url, podcast_id):
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                content = await response.text()
                
                # Generate a safe filename
                parsed_url = urlparse(url)
                filename = f"{podcast_id}_{os.path.basename(parsed_url.path)}"
                if not filename.endswith('.xml'):
                    filename += '.xml'
                
                # Save the RSS content
                filepath = os.path.join(RSS_BACKUP_DIR, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                return filepath
            else:
                print(f"Error downloading RSS for podcast {podcast_id}: HTTP {response.status}")
                return None
    except Exception as e:
        print(f"Error downloading RSS for podcast {podcast_id}: {str(e)}")
        return None

async def download_all_rss(podcasts):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for podcast in podcasts:
            podcast_id = podcast[0]
            url = podcast[4]
            tasks.append(download_rss(session, url, podcast_id))
        
        # Run all downloads concurrently with a limit of 10 concurrent requests
        results = []
        for i in range(0, len(tasks), 10):
            batch = tasks[i:i+10]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)
            # Small delay between batches
            await asyncio.sleep(0.5)
        
        return results

def main():
    conn = sqlite3.connect("podcastindex_feeds.db")
    cursor = conn.cursor()

    # First check what columns are available
    cursor.execute("PRAGMA table_info(podcasts)")
    columns = cursor.fetchall()
    print("Available columns:")
    for col in columns:
        print(f"- {col[1]}")  # Column name is at index 1

    # Calculate date 2 months ago
    two_months_ago = datetime.now() - timedelta(days=60)
    two_months_ago_timestamp = int(two_months_ago.timestamp())

    # Get recent Czech podcasts
    cursor.execute("""
        SELECT id, title, language, description, url, lastupdate
        FROM podcasts 
        WHERE language IN ('cs', 'cs-CS')
        AND lastupdate >= ?
        ORDER BY lastupdate DESC
    """, (two_months_ago_timestamp,))

    # Fetch all results
    czech_podcasts = cursor.fetchall()

    # Run async downloads
    print("\nDownloading RSS feeds...")
    rss_paths = asyncio.run(download_all_rss(czech_podcasts))

    # Convert to list of dictionaries
    podcasts_list = []
    for podcast, rss_path in zip(czech_podcasts, rss_paths):
        last_update = datetime.fromtimestamp(podcast[5]).strftime('%Y-%m-%d %H:%M:%S')
        podcast_data = {
            'id': podcast[0],
            'title': podcast[1],
            'language': podcast[2],
            'description': podcast[3],
            'url': podcast[4],
            'last_update': last_update,
            'rss_backup_path': rss_path
        }
        podcasts_list.append(podcast_data)

    # Write to JSON file
    with open('recent_czech_podcasts.json', 'w', encoding='utf-8') as f:
        json.dump(podcasts_list, f, ensure_ascii=False, indent=2)

    print(f"\nFound {len(podcasts_list)} Czech podcasts updated in the last 2 months")
    print("Top 10 most recently updated podcasts:")
    for podcast in podcasts_list[:10]:
        print(f"\nTitle: {podcast['title']}")
        print(f"Last update: {podcast['last_update']}")
        print(f"URL: {podcast['url']}")
        print(f"RSS backup: {podcast['rss_backup_path']}")

    print("\nComplete data has been saved to recent_czech_podcasts.json")
    print(f"RSS feeds have been saved to the {RSS_BACKUP_DIR} directory")

    conn.close()

if __name__ == "__main__":
    main() 