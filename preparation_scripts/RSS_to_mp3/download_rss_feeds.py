import json
import aiohttp
import asyncio
import os
from urllib.parse import urlparse
from datetime import datetime

# Create directory for RSS backups if it doesn't exist
RSS_BACKUP_DIR = "rss_by_language"
if not os.path.exists(RSS_BACKUP_DIR):
    os.makedirs(RSS_BACKUP_DIR)

async def download_rss(session, url, podcast_id, language_code):
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                content = await response.text()
                
                # Create language-specific directory
                lang_dir = os.path.join(RSS_BACKUP_DIR, language_code)
                if not os.path.exists(lang_dir):
                    os.makedirs(lang_dir)
                
                # Generate a safe filename
                parsed_url = urlparse(url)
                base_name = os.path.basename(parsed_url.path)
                # Remove any file extension if present
                base_name = os.path.splitext(base_name)[0]
                # Create filename in format LANG_ID_RSS_NAME.xml
                filename = f"{language_code}_{podcast_id}_{base_name}.xml"
                
                # Save the RSS content
                filepath = os.path.join(lang_dir, filename)
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
        for lang, lang_data in podcasts.items():
            for podcast in lang_data['podcasts']:
                tasks.append(download_rss(
                    session, 
                    podcast['url'], 
                    podcast['id'],
                    podcast['language_code']
                ))
        
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
    # Load the JSON file
    try:
        with open('multilingual_podcasts_rss.json', 'r', encoding='utf-8') as f:
            rss_data = json.load(f)
    except FileNotFoundError:
        print("Error: multilingual_podcasts_rss.json not found. Please run extract_recent_multilingual_podcasts.py first.")
        return
    except json.JSONDecodeError:
        print("Error: Invalid JSON file. Please check the file format.")
        return

    print(f"Starting download of RSS feeds for {rss_data['total_languages']} languages...")
    print(f"Total podcasts to download: {sum(len(data['podcasts']) for data in rss_data['languages'].values())}")

    # Download all RSS feeds
    results = asyncio.run(download_all_rss(rss_data['languages']))

    # Count successful downloads
    successful_downloads = len([r for r in results if r is not None])
    print(f"\nDownload complete!")
    print(f"Successfully downloaded: {successful_downloads} RSS feeds")
    print(f"Failed downloads: {len(results) - successful_downloads}")
    print(f"\nRSS feeds have been saved to the {RSS_BACKUP_DIR} directory, organized by language code.")

if __name__ == "__main__":
    main() 