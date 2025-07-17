import os
import json
from pathlib import Path

def count_underscores_in_file(file_path):
    """Count underscores in a JSON file's lemmas."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        underscore_count = 0
        total_words = 0
        
        for word in data.get('words', []):
            total_words += 1
            lemma = word.get('lemma', '')
            if lemma:
                underscore_count += lemma.count('_')
        
        return underscore_count, total_words
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return 0, 0

def main():
    # Path to transcriptions directory
    transcriptions_dir = Path("../transcriptions")
    
    # Get all lemmatized files
    lemmatized_files = list(transcriptions_dir.glob("*_lemmatized.json"))
    
    total_underscores = 0
    total_words = 0
    files_with_underscores = 0
    
    print("\nAnalyzing lemmatized files for underscore occurrences...")
    print("-" * 50)
    
    for file_path in lemmatized_files:
        underscores, words = count_underscores_in_file(file_path)
        if underscores > 0:
            files_with_underscores += 1
            print(f"\n{file_path.name}:")
            print(f"  Underscores: {underscores}")
            print(f"  Total words: {words}")
            print(f"  Underscore ratio: {underscores/words:.2%}")
        
        total_underscores += underscores
        total_words += words
    
    print("\nSummary:")
    print("-" * 50)
    print(f"Total files analyzed: {len(lemmatized_files)}")
    print(f"Files containing underscores: {files_with_underscores}")
    print(f"Total underscores found: {total_underscores}")
    print(f"Total words processed: {total_words}")
    if total_words > 0:
        print(f"Overall underscore ratio: {total_underscores/total_words:.2%}")

if __name__ == "__main__":
    main() 