import ufal.udpipe
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')

# Dictionary mapping ISO‐language codes to corresponding UDPipe model file paths.
LANG_MODEL_PATHS = {
    'en': os.path.join(MODELS_DIR, 'english-ud-1.2-160523.udpipe'),
    'fr': os.path.join(MODELS_DIR, 'french-ud-1.2-160523.udpipe'),
    'de': os.path.join(MODELS_DIR, 'german-ud-1.2-160523.udpipe'),
    'es': os.path.join(MODELS_DIR, 'spanish-ud-1.2-160523.udpipe'),
    'it': os.path.join(MODELS_DIR, 'italian-ud-1.2-160523.udpipe'),
    'nl': os.path.join(MODELS_DIR, 'dutch-ud-1.2-160523.udpipe'),
    'pt': os.path.join(MODELS_DIR, 'portuguese-ud-1.2-160523.udpipe'),
    'cs': os.path.join(MODELS_DIR, 'czech-ud-1.2-160523.udpipe'),
    'pl': os.path.join(MODELS_DIR, 'polish-ud-1.2-160523.udpipe'),
    'sv': os.path.join(MODELS_DIR, 'swedish-ud-1.2-160523.udpipe'),
    'da': os.path.join(MODELS_DIR, 'danish-ud-1.2-160523.udpipe'),
    'fi': os.path.join(MODELS_DIR, 'finnish-ud-1.2-160523.udpipe'),
    'ar': os.path.join(MODELS_DIR, 'arabic-ud-1.2-160523.udpipe'),
    'hi': os.path.join(MODELS_DIR, 'hindi-ud-1.2-160523.udpipe'),
    'no': os.path.join(MODELS_DIR, 'norwegian-ud-1.2-160523.udpipe'),
    'fa': os.path.join(MODELS_DIR, 'persian-ud-1.2-160523.udpipe'),
    'id': os.path.join(MODELS_DIR, 'indonesian-ud-1.2-160523.udpipe'),
    'bg': os.path.join(MODELS_DIR, 'bulgarian-ud-1.2-160523.udpipe'),
    'sl': os.path.join(MODELS_DIR, 'slovenian-ud-1.2-160523.udpipe'),
    'hr': os.path.join(MODELS_DIR, 'croatian-ud-1.2-160523.udpipe'),
    'eu': os.path.join(MODELS_DIR, 'basque-ud-1.2-160523.udpipe'),
    'el': os.path.join(MODELS_DIR, 'greek-ud-1.2-160523.udpipe'),
    'la': os.path.join(MODELS_DIR, 'latin-ud-1.2-160523.udpipe'),
    'got': os.path.join(MODELS_DIR, 'gothic-ud-1.2-160523.udpipe'),
    'cu': os.path.join(MODELS_DIR, 'old-church-slavonic-ud-1.2-160523.udpipe'),
    'ga': os.path.join(MODELS_DIR, 'irish-ud-1.2-160523.udpipe'),
    'hu': os.path.join(MODELS_DIR, 'hungarian-ud-1.2-160523.udpipe'),
    'ta': os.path.join(MODELS_DIR, 'tamil-ud-1.2-160523.udpipe'),
    'ro': os.path.join(MODELS_DIR, 'romanian-ud-1.2-160523.udpipe'),
    'et': os.path.join(MODELS_DIR, 'estonian-ud-1.2-160523.udpipe'),
    'he': os.path.join(MODELS_DIR, 'hebrew-ud-1.2-160523.udpipe'),
}

# Cache for the current language's pipeline
current_pipeline = None
current_language = None

def extract_lemma(conllu_output):
    """Extract lemma from CoNLL-U format output"""
    for line in conllu_output.split('\n'):
        if not line or line.startswith('#'):
            continue
        cols = line.split('\t')
        if len(cols) >= 3:
            return cols[2]
    return None

def get_pipeline(lang_code):
    """
    Get or create a pipeline for the given language code.
    Clears the cache if the language changes.
    """
    global current_pipeline, current_language
    
    # If language changed, clear the cache
    if current_language != lang_code:
        current_pipeline = None
        current_language = lang_code
    
    # If we have a cached pipeline, return it
    if current_pipeline is not None:
        return current_pipeline
    
    # Get model path
    model_path = LANG_MODEL_PATHS.get(lang_code)
    if not model_path or not os.path.isfile(model_path):
        return None
    
    try:
        # Load model
        model = ufal.udpipe.Model.load(model_path)
        if model is None:
            return None
        
        # Create pipeline
        pipeline = ufal.udpipe.Pipeline(
            model,
            'tokenize',
            'pos',
            'lemma',
            'conllu'
        )
        
        # Cache the pipeline
        current_pipeline = pipeline
        return pipeline
        
    except Exception as e:
        logger.error(f"Error creating pipeline for {lang_code}: {str(e)}")
        return None

def lemmatize(word, lang_code):
    """
    Lemmatize a single word using UDPipe.
    Returns the original word if lemmatization fails.
    """
    if not word or not isinstance(word, str):
        return word
    
    # Get pipeline for the language
    pipeline = get_pipeline(lang_code)
    if pipeline is None:
        return word
    
    try:
        # Process the word with a period to make it a proper sentence
        result = pipeline.process(f"{word}.")
        lemma = extract_lemma(result)
        return lemma if lemma else word
        
    except Exception as e:
        logger.error(f"Error lemmatizing word '{word}' in {lang_code}: {str(e)}")
        return word

if __name__ == "__main__":
    # Test the lemmatizer
    test_words = [
        "running",
        "went",
        "better",
        "happiest",
        "studying"
    ]
    
    print("\nInput word -> Lemmatized word")
    print("-" * 30)
    
    for word in test_words:
        lemma = lemmatize(word, 'en')
        print(f"{word:15} -> {lemma}")
