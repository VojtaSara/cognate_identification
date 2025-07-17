import sqlite3

def read_podcast_feed():
    """Read one feed entry from the podcastindex_feeds database."""
    try:
        # Connect to the database
        conn = sqlite3.connect('podcastindex_feeds.db')
        cursor = conn.cursor()
        
        # Get the table schema
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("\nTables in database:")
        for table in tables:
            print(f"- {table[0]}")
            
            # Get column names for each table
            cursor.execute(f"PRAGMA table_info({table[0]})")
            columns = cursor.fetchall()
            print("  Columns:")
            for col in columns:
                print(f"  - {col[1]} ({col[2]})")
        
        # Read one row from the feeds table
        cursor.execute("SELECT * FROM feeds LIMIT 1")
        row = cursor.fetchone()
        
        if row:
            print("\nSample feed entry:")
            print("-" * 50)
            # Get column names
            cursor.execute("PRAGMA table_info(feeds)")
            columns = [col[1] for col in cursor.fetchall()]
            
            # Print each field
            for col, val in zip(columns, row):
                print(f"{col:20}: {val}")
        else:
            print("No feeds found in the database")
            
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    read_podcast_feed() 