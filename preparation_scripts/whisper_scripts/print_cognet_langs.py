def print_cognet_languages():
    """Print all unique languages found in COGNET file."""
    languages = set()
    
    try:
        with open("PodcastRealDataset/wordleveltranscripts/CogNet-v2.0.tsv", 'r', encoding='utf-8') as f:
            # Skip header line
            next(f)
            
            for line in f:
                fields = line.strip().split('\t')
                # Skip empty lines
                if len(fields) < 3:
                    continue
                    
                # Process each language-word pair
                for i in range(1, len(fields), 2):
                    if i + 1 < len(fields):
                        lang = fields[i].strip().lower()
                        if lang and lang != 'lang':  # Skip empty and header-like values
                            languages.add(lang)
        
        print("\nLanguages in COGNET:")
        print("-" * 30)
        for lang in sorted(languages):
            print(lang)
            
    except Exception as e:
        print(f"Error reading COGNET file: {str(e)}")

if __name__ == "__main__":
    print_cognet_languages() 