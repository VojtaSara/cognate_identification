import os
from pathlib import Path
import librosa
import numpy as np
from tqdm import tqdm
import time

def get_audio_duration(file_path):
    """Get duration of audio file in seconds."""
    try:
        duration = librosa.get_duration(path=file_path)
        return duration
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def analyze_audio_lengths():
    """Analyze audio file lengths across all language folders."""
    base_dir = Path("rss_by_language")
    all_durations = []
    language_stats = {}
    
    # Get all language folders
    language_folders = [d for d in base_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(language_folders)} language folders")
    print("Analyzing audio files...")
    
    start_time = time.time()
    
    # Process each language folder
    for lang_folder in tqdm(language_folders, desc="Processing languages"):
        lang = lang_folder.name
        durations = []
        
        # Get all audio files in the language folder
        audio_files = list(lang_folder.rglob("*.mp3")) + list(lang_folder.rglob("*.wav"))
        
        # Process each audio file
        for audio_file in tqdm(audio_files, desc=f"Processing {lang}", leave=False):
            duration = get_audio_duration(str(audio_file))
            if duration is not None:
                durations.append(duration)
                all_durations.append(duration)
        
        # Calculate statistics for this language
        if durations:
            language_stats[lang] = {
                'count': len(durations),
                'min': min(durations),
                'max': max(durations),
                'avg': np.mean(durations),
                'total_duration': sum(durations)
            }
    
    # Calculate overall statistics
    if all_durations:
        overall_stats = {
            'total_files': len(all_durations),
            'min': min(all_durations),
            'max': max(all_durations),
            'avg': np.mean(all_durations),
            'total_duration': sum(all_durations)
        }
        
        print("\nOverall Statistics:")
        print("-" * 50)
        print(f"Total audio files: {overall_stats['total_files']:,}")
        print(f"Total duration: {overall_stats['total_duration']/3600:.1f} hours")
        print(f"Average length: {overall_stats['avg']/60:.1f} minutes")
        print(f"Shortest file: {overall_stats['min']/60:.1f} minutes")
        print(f"Longest file: {overall_stats['max']/60:.1f} minutes")
        
        print("\nLanguage-specific Statistics:")
        print("-" * 50)
        # Sort languages by number of files
        sorted_langs = sorted(language_stats.items(), key=lambda x: x[1]['count'], reverse=True)
        
        for lang, stats in sorted_langs:
            print(f"\n{lang.upper()}:")
            print(f"  Files: {stats['count']:,}")
            print(f"  Total duration: {stats['total_duration']/3600:.1f} hours")
            print(f"  Average length: {stats['avg']/60:.1f} minutes")
            print(f"  Shortest: {stats['min']/60:.1f} minutes")
            print(f"  Longest: {stats['max']/60:.1f} minutes")
    
    print(f"\nAnalysis completed in {time.time() - start_time:.1f} seconds")

if __name__ == "__main__":
    analyze_audio_lengths() 