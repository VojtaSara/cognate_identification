import ufal.udpipe
import os
import sys
import traceback
import json
from pathlib import Path

# Mapping of language codes to UDPipe model names
LANGUAGE_MODELS = {
    'ar': 'arabic-ud-1.2-160523.udpipe',      # Arabic
    'bg': 'bulgarian-ud-1.2-160523.udpipe',   # Bulgarian
    'hr': 'croatian-ud-1.2-160523.udpipe',    # Croatian
    'cs': 'czech-ud-1.2-160523.udpipe',       # Czech
    'da': 'danish-ud-1.2-160523.udpipe',      # Danish
    'nl': 'dutch-ud-1.2-160523.udpipe',       # Dutch
    'en': 'english-ud-1.2-160523.udpipe',     # English
    'et': 'estonian-ud-1.2-160523.udpipe',    # Estonian
    'fi': 'finnish-ud-1.2-160523.udpipe',     # Finnish
    'fr': 'french-ud-1.2-160523.udpipe',      # French
    'de': 'german-ud-1.2-160523.udpipe',      # German
    'el': 'greek-ud-1.2-160523.udpipe',       # Greek
    'he': 'hebrew-ud-1.2-160523.udpipe',      # Hebrew
    'hi': 'hindi-ud-1.2-160523.udpipe',       # Hindi
    'hu': 'hungarian-ud-1.2-160523.udpipe',   # Hungarian
    'id': 'indonesian-ud-1.2-160523.udpipe',  # Indonesian
    'ga': 'irish-ud-1.2-160523.udpipe',       # Irish
    'it': 'italian-ud-1.2-160523.udpipe',     # Italian
    'la': 'latin-ud-1.2-160523.udpipe',       # Latin
    'no': 'norwegian-ud-1.2-160523.udpipe',   # Norwegian
    'fa': 'persian-ud-1.2-160523.udpipe',     # Persian
    'pl': 'polish-ud-1.2-160523.udpipe',      # Polish
    'pt': 'portuguese-ud-1.2-160523.udpipe',  # Portuguese
    'ro': 'romanian-ud-1.2-160523.udpipe',    # Romanian
    'sl': 'slovenian-ud-1.2-160523.udpipe',   # Slovenian
    'es': 'spanish-ud-1.2-160523.udpipe',     # Spanish
    'sv': 'swedish-ud-1.2-160523.udpipe',     # Swedish
    'ta': 'tamil-ud-1.2-160523.udpipe',       # Tamil
    'eu': 'basque-ud-1.2-160523.udpipe',      # Basque
}

# Ancient language models
ANCIENT_MODELS = {
    'grc': 'ancient-greek-ud-1.2-160523.udpipe',        # Ancient Greek
    'grc-proiel': 'ancient-greek-proiel-ud-1.2-160523.udpipe',  # Ancient Greek (PROIEL)
    'got': 'gothic-ud-1.2-160523.udpipe',               # Gothic
    'la-itt': 'latin-itt-ud-1.2-160523.udpipe',         # Latin (ITTB)
    'la-proiel': 'latin-proiel-ud-1.2-160523.udpipe',   # Latin (PROIEL)
    'cu': 'old-church-slavonic-ud-1.2-160523.udpipe',   # Old Church Slavonic
}

# Additional models
ADDITIONAL_MODELS = {
    'fi-ftb': 'finnish-ftb-ud-1.2-160523.udpipe',  # Finnish (FTB)
}

# Combine all mappings
ALL_MODELS = {**LANGUAGE_MODELS, **ANCIENT_MODELS, **ADDITIONAL_MODELS}

def print_debug(msg):
    """Print debug message with timestamp"""
    print(f"[DEBUG] {msg}")

def get_model_path(lang_code: str) -> str:
    """Get the UDPipe model path for a given language code."""
    # Try to get the model name directly from our mappings
    model_name = ALL_MODELS.get(lang_code.lower())
    if model_name:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', model_name)
    return None

def extract_lemma(conllu_output):
    """Extract lemma from CoNLL-U format output"""
    for line in conllu_output.split('\n'):
        if not line or line.startswith('#'):
            continue
        cols = line.split('\t')
        if len(cols) >= 3:
            return cols[2]
    return None

def process_transcription_file(input_file: str, output_file: str, lang_code: str = 'english'):
    """
    Process a transcription file and add lemmatization to each word.
    """
    try:
        # Load the transcription file
        print_debug(f"Loading transcription file: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        language = data['metadata']['language']
        print_debug(f"Detected language code: {language}")

        # Get model path
        model_path = get_model_path(language)
        if not model_path:
            print_debug(f"No model found for language: {language}")
            return None
        
        print_debug(f"Loading model from: {model_path}")
        model = ufal.udpipe.Model.load(model_path)
        if model is None:
            print_debug("Failed to load model!")
            return
            
        pipeline = ufal.udpipe.Pipeline(
            model,
            'tokenize',
            'pos',
            'lemma',
            'conllu'
        )
        
        # Process each word
        print_debug("Processing words...")
        for word in data['words']:
            try:
                original_word = word['word']
                # Process the word with a period to make it a proper sentence
                result = pipeline.process(f"{original_word}.")
                lemma = extract_lemma(result)
                if lemma:
                    word['lemma'] = lemma
                else:
                    word['lemma'] = original_word
            except Exception as e:
                print_debug(f"Error processing word '{word.get('word', 'UNKNOWN')}': {str(e)}")
                word['lemma'] = word['word']
        
        # Save the updated file
        print_debug(f"Saving updated transcription to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print_debug("Processing complete!")
        
    except Exception as e:
        print_debug(f"Fatal error: {str(e)}")
        print_debug(traceback.format_exc())

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # If file path provided, process the file
        input_file = sys.argv[1]
        output_file = input_file.replace('.json', '_lemmatized.json')
        process_transcription_file(input_file, output_file)