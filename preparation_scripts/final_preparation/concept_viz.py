import pandas as pd
import networkx as nx
from typing import List, Tuple, Set
import time

def analyze_problematic_rows(df: pd.DataFrame) -> None:
    """Analyze and print examples of rows that would be skipped."""
    print("\nAnalyzing data for potential problems...")
    
    # Check for actual NaN values (not the string 'nan')
    nan_counts = df[['lang 1', 'word 1', 'lang 2', 'word 2']].apply(
        lambda x: x.isna() & (x != 'nan')
    ).sum()
    print("\nActual NaN counts in each column (excluding 'nan' language code):")
    print(nan_counts)
    
    # Check for float values (excluding 'nan' language code)
    float_rows = df[
        (df['lang 1'].apply(lambda x: isinstance(x, float) and not pd.isna(x))) |
        (df['word 1'].apply(lambda x: isinstance(x, float) and not pd.isna(x))) |
        (df['lang 2'].apply(lambda x: isinstance(x, float) and not pd.isna(x))) |
        (df['word 2'].apply(lambda x: isinstance(x, float) and not pd.isna(x)))
    ]
    
    if not float_rows.empty:
        print("\nFound rows with float values (first 5 examples):")
        print(float_rows[['lang 1', 'word 1', 'lang 2', 'word 2']].head())
    
    # Check for rows where both pairs are invalid (considering 'nan' as valid)
    invalid_pairs = df[
        ((df['lang 1'].isna() | (df['word 1'].isna() & (df['lang 1'] != 'nan'))) & 
         (df['lang 2'].isna() | (df['word 2'].isna() & (df['lang 2'] != 'nan'))))
    ]
    
    if not invalid_pairs.empty:
        print("\nFound rows where both language-word pairs are invalid (first 5 examples):")
        print(invalid_pairs[['lang 1', 'word 1', 'lang 2', 'word 2']].head())
    
    print(f"\nTotal rows that would be skipped: {len(invalid_pairs)}")
    print(f"Percentage of total rows: {(len(invalid_pairs) / len(df) * 100):.2f}%")

def get_concept_components() -> List[Tuple[str, Set[Tuple[str, str]]]]:
    """
    Returns a list of tuples containing (concept_id, set of (language, word) tuples)
    for each connected component in each concept.
    """
    start_time = time.time()
    
    # Define only the columns we care about
    columns = ["concept id", "lang 1", "word 1", "lang 2", "word 2", "translit 1", "translit 2"]

    # Read only first 7 columns, skip any extras
    print("Reading CSV...")
    df = pd.read_csv("CogNet-v2.0.tsv", sep="\t", names=columns, usecols=range(len(columns)), skiprows=1)
    print(f"CSV read time: {time.time() - start_time:.2f} seconds")
    
    # Analyze problematic rows before processing
    analyze_problematic_rows(df)

    # Pre-split the data by concept_id
    print("\nSplitting data by concept...")
    split_start = time.time()
    concept_groups = {concept_id: group for concept_id, group in df.groupby("concept id")}
    print(f"Data split time: {time.time() - split_start:.2f} seconds")
    print(f"Found {len(concept_groups)} unique concepts")

    # List to store all components
    all_components = []
    process_start = time.time()
    total_components = 0
    skipped_rows = 0

    # Process each concept
    for concept_id, filtered in concept_groups.items():
        # Create graph
        G = nx.Graph()
        
        # Add nodes and edges
        for _, row in filtered.iterrows():
            try:
                # Handle 'nan' as a valid language code
                lang1 = str(row['lang 1']).strip() if pd.notna(row['lang 1']) or row['lang 1'] == 'nan' else ""
                word1 = str(row['word 1']).strip() if pd.notna(row['word 1']) else ""
                lang2 = str(row['lang 2']).strip() if pd.notna(row['lang 2']) or row['lang 2'] == 'nan' else ""
                word2 = str(row['word 2']).strip() if pd.notna(row['word 2']) else ""
                
                # Skip only if both pairs are invalid
                if (not lang1 or not word1) and (not lang2 or not word2):
                    skipped_rows += 1
                    continue
                
                # Add valid pairs to the graph
                if lang1 and word1:
                    G.add_node((lang1, word1))
                if lang2 and word2:
                    G.add_node((lang2, word2))
                if lang1 and word1 and lang2 and word2:
                    G.add_edge((lang1, word1), (lang2, word2))
                    
            except Exception as e:
                print(f"Error processing row in concept {concept_id}: {e}")
                print(f"Row data: {row.to_dict()}")
                skipped_rows += 1
                continue
        
        # Get all connected components
        components = list(nx.connected_components(G))
        
        # Add each component to our list
        for component in components:
            all_components.append((concept_id, component))
            total_components += 1
        
        if len(all_components) % 100 == 0:
            elapsed = time.time() - process_start
            print(f"Processed {len(all_components)} components in {elapsed:.2f} seconds")
            print(f"Skipped {skipped_rows} invalid rows so far")

    total_time = time.time() - start_time
    print(f"\nFinal Statistics:")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Total components found: {total_components}")
    print(f"Total rows skipped: {skipped_rows}")
    print(f"Average time per component: {total_time/total_components:.4f} seconds")
    
    return all_components

if __name__ == "__main__":
    components = get_concept_components()
    print(f"Total number of components across all concepts: {len(components)}")

