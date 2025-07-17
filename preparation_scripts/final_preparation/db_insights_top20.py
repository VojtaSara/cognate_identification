import sqlite3

def print_separator(title):
    print("\n" + "="*50)
    print(f" {title} ".center(50, "="))
    print("="*50 + "\n")

def get_preferred_translation(cursor, component_id):
    """Get English, Czech, or French translation if available."""
    cursor.execute("""
        SELECT language, text
        FROM words
        WHERE component_id = ?
        AND language IN ('eng', 'ces', 'fra')
        ORDER BY CASE language
            WHEN 'eng' THEN 1
            WHEN 'ces' THEN 2
            WHEN 'fra' THEN 3
        END
        LIMIT 1;
    """, (component_id,))
    return cursor.fetchone()

def print_component_summary(cursor, component_id, concept_id, word_count, rank):
    # Get preferred translation
    translation = get_preferred_translation(cursor, component_id)
    
    # Get all languages in this component
    cursor.execute("""
        SELECT DISTINCT language
        FROM words
        WHERE component_id = ?
        ORDER BY language;
    """, (component_id,))
    languages = [lang[0] for lang in cursor.fetchall()]
    
    # Print component info
    print(f"\n{rank}. Component: {component_id}")
    print(f"   Concept: {concept_id}")
    print(f"   Words: {word_count}")
    print(f"   Languages ({len(languages)}): {', '.join(languages)}")
    if translation:
        lang, text = translation
        print(f"   Translation ({lang}): {text}")
    print("-" * 50)

def main():
    # Connect to the database
    conn = sqlite3.connect('cognates_2.db')
    cursor = conn.cursor()
    
    # Get the top 20 largest cognate components
    print_separator("TOP 20 COGNATE COMPONENTS")
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
        SELECT component_id, concept_id, word_count, rank
        FROM RankedComponents
        WHERE rank <= 20
        ORDER BY rank;
    """)
    
    components = cursor.fetchall()
    
    if components:
        for component in components:
            component_id = component[0]
            concept_id = component[1]
            word_count = component[2]
            rank = component[3]
            print_component_summary(cursor, component_id, concept_id, word_count, rank)
    else:
        print("No components found in the database.")
    
    # Print some statistics
    cursor.execute("""
        SELECT 
            COUNT(DISTINCT cg.component_id) as total_components,
            COUNT(DISTINCT cg.concept_id) as total_concepts,
            COUNT(DISTINCT w.language) as total_languages,
            COUNT(w.word_id) as total_words
        FROM cognate_groups cg
        LEFT JOIN words w ON cg.component_id = w.component_id;
    """)
    stats = cursor.fetchone()
    
    print_separator("DATABASE STATISTICS")
    print(f"Total Components: {stats[0]}")
    print(f"Total Concepts: {stats[1]}")
    print(f"Total Languages: {stats[2]}")
    print(f"Total Words: {stats[3]}")
    
    conn.close()

if __name__ == "__main__":
    main() 