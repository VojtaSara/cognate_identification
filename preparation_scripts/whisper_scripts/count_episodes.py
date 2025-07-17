import sqlite3
import time

def count_total_episodes():
    """Count total episodes by summing episodeCount from all podcasts."""
    try:
        print("Connecting to database...")
        start_time = time.time()
        
        # Connect to the database
        conn = sqlite3.connect('PodcastRealDataset/podcastindex_feeds.db')
        cursor = conn.cursor()
        
        print("Counting total podcasts...")
        # Get total number of podcasts
        cursor.execute("SELECT COUNT(*) FROM podcasts")
        total_podcasts = cursor.fetchone()[0]
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        
        print("\nCounting total episodes...")
        start_time = time.time()
        # Get sum of episodeCount
        cursor.execute("SELECT SUM(episodeCount) FROM podcasts")
        total_episodes = cursor.fetchone()[0]
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        
        print("\nCalculating statistics...")
        start_time = time.time()
        # Get some statistics
        cursor.execute("""
            SELECT 
                MIN(episodeCount) as min_episodes,
                MAX(episodeCount) as max_episodes,
                AVG(episodeCount) as avg_episodes
            FROM podcasts
        """)
        stats = cursor.fetchone()
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        
        print("\nPodcast Statistics:")
        print("-" * 50)
        print(f"Total number of podcasts: {total_podcasts:,}")
        print(f"Total number of episodes: {total_episodes:,}")
        print(f"Minimum episodes per podcast: {stats[0]:,}")
        print(f"Maximum episodes per podcast: {stats[1]:,}")
        print(f"Average episodes per podcast: {stats[2]:,.1f}")
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    count_total_episodes() 