import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import time
from collections import deque

class WordsPerMinuteTracker:
    def __init__(self, window_size=6):  # 6 samples = 1 minute of data
        self.window_size = window_size
        self.rates = deque(maxlen=window_size)
        self.all_averages = []  # Store all calculated averages
        self.last_words = None
        self.last_time = None
    
    def update(self, current_words, current_time):
        if self.last_words is not None and self.last_time is not None:
            time_diff = (current_time - self.last_time).total_seconds() / 60  # Convert to minutes
            if time_diff > 0:
                words_diff = current_words - self.last_words
                current_rate = words_diff / time_diff
                self.rates.append(current_rate)
        
        self.last_words = current_words
        self.last_time = current_time
        
        if self.rates:
            current_avg = sum(self.rates) / len(self.rates)
            self.all_averages.append(current_avg)
            return current_avg
        return 0.0
    
    def get_overall_average(self):
        if not self.all_averages:
            return 0.0
        return sum(self.all_averages) / len(self.all_averages)

def analyze_transcriptions(wpm_tracker):
    # Get current timestamp
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Initialize counters
    word_counts = defaultdict(int)
    file_counts = defaultdict(int)
    total_words = 0
    
    # Process all JSON files in the transcriptions directory
    transcriptions_dir = Path("transcriptions")
    if not transcriptions_dir.exists():
        print(f"{timestamp} - No transcriptions directory found")
        return
    
    for json_file in transcriptions_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                language = data['metadata']['language']
                word_count = len(data['words'])
                word_counts[language] += word_count
                file_counts[language] += 1
                total_words += word_count
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    # Print results in a concise format
    print(f"\n{timestamp}")
    print("-" * 50)
    print(f"{'Language':<8} {'Files':<8} {'Words':<12} {'Avg Words/File':<15}")
    print("-" * 50)
    
    # Sort by word count in descending order
    for lang, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True):
        avg_words = count / file_counts[lang]
        print(f"{lang:<8} {file_counts[lang]:<8} {count:<12} {avg_words:>8.1f}")
    
    # Print total summary
    print("-" * 50)
    print(f"TOTAL WORDS: {total_words:,}")
    
    # Update and print both averages
    current_avg = wpm_tracker.update(total_words, current_time)
    overall_avg = wpm_tracker.get_overall_average()
    print(f"AVG WORDS PER MINUTE (1min): {current_avg:.1f}")
    print(f"OVERALL AVG WPM: {overall_avg:.1f}")

def main():
    print("Starting continuous analysis (press Ctrl+C to stop)...")
    wpm_tracker = WordsPerMinuteTracker()
    
    try:
        while True:
            analyze_transcriptions(wpm_tracker)
            time.sleep(10)  # Wait for 10 seconds
    except KeyboardInterrupt:
        print("\nAnalysis stopped by user")

if __name__ == "__main__":
    main() 