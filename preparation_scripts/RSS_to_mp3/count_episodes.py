import sqlite3

def count_total_episodes():
    """Count total episodes by summing episodeCount from all podcasts."""
    try:
        # Connect to the database
        conn = sqlite3.connect('podcastindex_feeds.db')
        cursor = conn.cursor()
        
        # Get total number of feeds
        cursor.execute("SELECT COUNT(*) FROM podcasts")
        total_podcasts = cursor.fetchone()[0]
        
        # Get sum of episodeCount
        cursor.execute("SELECT SUM(episodeCount) FROM podcasts")
        total_episodes = cursor.fetchone()[0]
        
        # Get some statistics
        cursor.execute("""
            SELECT 
                MIN(episodeCount) as min_episodes,
                MAX(episodeCount) as max_episodes,
                AVG(episodeCount) as avg_episodes
            FROM podcasts
        """)
        stats = cursor.fetchone()
        
        print("\nPodcast Feed Statistics:")
        print("-" * 50)
        print(f"Total number of podcasts: {total_podcasts:,}")
        print(f"Total number of episodes: {total_episodes:,}")
        print(f"Minimum episodes per feed: {stats[0]:,}")
        print(f"Maximum episodes per feed: {stats[1]:,}")
        print(f"Average episodes per feed: {stats[2]:,.1f}")
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    count_total_episodes() 