import os
import subprocess
import glob
from pathlib import Path

def process_files(input_dir, pattern="*.json"):
    """
    Process all JSON files in the input directory using test_udpipe.py
    Skips files that already have _lemmatized in their name
    """
    # Get all JSON files in the directory
    input_files = glob.glob(os.path.join(input_dir, pattern))
    
    # Filter out already lemmatized files
    input_files = [f for f in input_files if "_lemmatized" not in f]
    
    print(f"Found {len(input_files)} files to process")
    
    # Process each file
    for input_file in input_files:
        print(f"\nProcessing: {input_file}")
        try:
            # Run test_udpipe.py with the input file
            result = subprocess.run(
                ["python", "WhisperVault/lemmatize/test_udpipe.py", input_file],
                capture_output=True,
                text=True
            )
            
            # Print output
            if result.stdout:
                print("Output:", result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)
                
            print(f"Completed processing: {input_file}")

            
        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")


if __name__ == "__main__":
    # Get the directory where this script is located	
    this_dir = os.path.dirname(os.path.abspath(__file__))
    input_directory = os.path.join(this_dir, "..", "transcriptions")
    print(f"Processing files in: {input_directory}")
    process_files(input_directory) 