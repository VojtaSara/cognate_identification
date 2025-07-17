import sqlite3
import sys

def print_separator(title):
    print("\n" + "="*50)
    print(f" {title} ".center(50, "="))
    print("="*50 + "\n")

def print_component_details(cursor, component_id, concept_id, word_count):
    print(f"\nComponent: {component_id}")
    print(f"Concept: {concept_id}")
    print(f"Contains {word_count} words")
    print("-" * 50)
    
    # Get all words from this component, ordered by language
    cursor.execute("""
        SELECT language, text
        FROM words
        WHERE component_id = ?
        ORDER BY language, text;
    """, (component_id,))
    
    words = cursor.fetchall()
    current_lang = None
    
    for word in words:
        lang, text = word
        if lang != current_lang:
            if current_lang is not None:
                print()  # Add blank line between languages
            current_lang = lang
            print(f"{lang}:", end=" ")
        else:
            print(",", end=" ")
        print(text, end="")
    print("\n")

def main():
    # Check if rank argument is provided
    if len(sys.argv) != 2:
        print("Usage: python db_insights.py <rank>")
        print("Example: python db_insights.py 3  # Shows the 3rd largest component")
        sys.exit(1)
    
    try:
        rank = int(sys.argv[1])
        if rank < 1:
            raise ValueError("Rank must be positive")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please provide a positive integer for the rank")
        sys.exit(1)
    
    # Connect to the database
    conn = sqlite3.connect('cognates_2.db')
    cursor = conn.cursor()
    
    # Get the nth largest cognate component
    print_separator(f"{rank}TH LARGEST COGNATE COMPONENT")
    cursor.execute("""
        WITH RankedComponents AS (
            SELECT 
                cg.component_id,
                cg.concept_id,
                COUNT(w.word_id) as word_count,
                ROW_NUMBER() OVER (ORDER BY COUNT(w.word_id) DESC) as rank
            FROM cognate_groups cg
            LEFT JOIN words w ON cg.component_id = w.component_id
            GROUP BY cg.component_id
        )
        SELECT component_id, concept_id, word_count
        FROM RankedComponents
        WHERE rank = ?;
    """, (rank,))
    
    component = cursor.fetchone()
    
    if component:
        component_id = component[0]
        concept_id = component[1]
        word_count = component[2]
        print_component_details(cursor, component_id, concept_id, word_count)
    else:
        print(f"No component found at rank {rank}.")
        # Show how many components exist
        cursor.execute("SELECT COUNT(DISTINCT component_id) FROM cognate_groups")
        total_components = cursor.fetchone()[0]
        print(f"Total components in database: {total_components}")
    
    conn.close()

if __name__ == "__main__":
    main() 