import json
from rich.console import Console
from rich.table import Table
from rich import box

def get_words_from_wordlevel(data):
    words = []
    for segment in data['segments']:
        if 'words' in segment:
            for word in segment['words']:
                words.append({
                    'word': word['word'],
                    'start': word['start'],
                    'end': word['end'],
                    'confidence': word.get('probability', None)
                })
    return words

def get_words_from_timestamped(data):
    return data['words']

def main():
    console = Console()
    
    # Load transcriptions
    with open('transcription_wordlevel.json', 'r', encoding='utf-8') as f:
        wordlevel = json.load(f)
    with open('transcription_20250517_185828.json', 'r', encoding='utf-8') as f:
        timestamped = json.load(f)
        
    # Get words from both formats
    wordlevel_words = get_words_from_wordlevel(wordlevel)
    timestamped_words = get_words_from_timestamped(timestamped)
    
    # Create tables
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Time", style="cyan")
    table.add_column("Wordlevel Word", style="green")
    table.add_column("Wordlevel Conf", style="yellow")
    table.add_column("", style="dim")  # Spacer
    table.add_column("Time", style="cyan")
    table.add_column("Timestamped Word", style="green")
    table.add_column("Timestamped Conf", style="yellow")
    
    # Add rows
    for word1, word2 in zip(wordlevel_words, timestamped_words):
        conf1 = f"{word1['confidence']:.3f}" if word1['confidence'] is not None else "N/A"
        conf2 = f"{word2['confidence']:.3f}" if word2['confidence'] is not None else "N/A"
        
        table.add_row(
            f"{word1['start']:.2f}s",
            word1['word'],
            conf1,
            "|",
            f"{word2['start']:.2f}s",
            word2['word'],
            conf2
        )
    
    # Print the table
    console.print(table)

if __name__ == "__main__":
    main() 