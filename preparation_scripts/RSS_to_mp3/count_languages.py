import json
from collections import Counter

# Read the languages from the JSON file
with open('languages.json', 'r', encoding='utf-8') as f:
    languages = json.load(f)

# Count occurrences of each language
language_counts = Counter(languages)

# Convert to dictionary for JSON serialization
language_stats = dict(language_counts)

# Write the counts to a new JSON file
with open('language_counts.json', 'w', encoding='utf-8') as f:
    json.dump(language_stats, f, ensure_ascii=False, indent=2)

# Print summary
print(f"Total unique languages: {len(language_stats)}")
print("\nTop 10 languages:")
for lang, count in sorted(language_stats.items(), key=lambda x: x[1], reverse=True)[:100]:
    print(f"{lang}: {count}")