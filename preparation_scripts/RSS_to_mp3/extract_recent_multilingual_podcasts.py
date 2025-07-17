import sqlite3
import json
import aiohttp
import asyncio
import os
from datetime import datetime, timedelta
from urllib.parse import urlparse
import re

# Create directory for RSS backups if it doesn't exist
RSS_BACKUP_DIR = "rss_backups"
if not os.path.exists(RSS_BACKUP_DIR):
    os.makedirs(RSS_BACKUP_DIR)

# Common language codes and their variations
LANGUAGE_MAPPING = {
    'en': ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU', 'ENG', 'Engli', 'Eng', 'ENU', 'ENH', 'english', 'English', 'ENGLI', 'en_EN', 'en_AU', 'en_CA', 'en_IE', 'en_JM', 'en_NZ', 'en_ZA', 'en-us', 'en-bz', 'en-ng', 'en-gh', 'en-ke', 'en-sg', 'en-tz', 'en-ug', 'en-zm', 'en-bw', 'en-cy', 'en-fi', 'en-fr', 'en-fr-us', 'en-gp', 'en-in', 'en-jp', 'en-mx', 'en-nl', 'en-se', 'en-tt', 'en-usa', 'en-eng', 'en-aus', 'en-cac', 'en-cdn', 'en-es', 'en-gb', 'en-modif', 'en-us,zh', 'en-us es', 'en-us-it', 'en-us,fa', 'en-fr-us', 'en-gb; a'],
    'es': ['es', 'es-ES', 'es-MX', 'es-AR', 'ES', 'español', 'espanol', 'spa-ES', 'spa-es', 'es_ES', 'es_US', 'es_VE', 'es_NI', 'es_PR', 'es_SV', 'es_419', 'es-us', 'es-ve', 'es-pr', 'es-sv', 'es-ec', 'es-gt', 'es-hn', 'es-pa', 'es-do', 'es-bo', 'es-py', 'es-ni', 'es-dr', 'es-spa', 'es-41', 'es-ca', 'es-la', 'es-sp', 'es-uy', 'es-CO', 'es-SPA', 'es-LA', 'es-SP', 'es-ES', 'es-US', 'es-VE', 'es-PR', 'es-SV', 'es-EC', 'es-GT', 'es-HN', 'es-PA', 'es-DO', 'es-BO', 'es-PY', 'es-NI', 'es-DR', 'es-SPA', 'es-41', 'es-CA', 'es-LA', 'es-SP', 'es-UY'],
    'fr': ['fr', 'fr-FR', 'fr-CA', 'fre', 'francais', 'French', 'French-f', 'fr-BE', 'fr-LU', 'fr-CH', 'fr-CI', 'fr-HT', 'fr-MU', 'fr-BF', 'fr-fr', 'fr-be', 'fr-lu', 'fr-ch', 'fr-ci', 'fr-ht', 'fr-mu', 'fr-bf', 'fr-fra', 'fr-frs', 'fr-fr, e', 'Fr'],
    'de': ['de', 'de-DE', 'de-AT', 'de-CH', 'DE', 'deutsch', 'De-DE', 'De-de', 'DE-de', 'ger', 'German', 'de-LU', 'de-li', 'de-lu', 'de-At', 'de_CH', 'de_AT', 'de-de, e', 'de-ch,gs'],
    'it': ['it', 'it-IT', 'IT', 'Italian', 'it-CH', 'it-ch'],
    'pt': ['pt', 'pt-BR', 'pt-PT', 'PT', 'Portuguê', 'pt_BR', 'pt-Br', 'pt-br', 'PT-br'],
    'ru': ['ru', 'ru-RU', 'RUS', 'rus', 'ru-mo', 'ru-by', 'ru-UA', 'ru-US', 'ru-Ru'],
    'ja': ['ja', 'ja-JP', 'JA', 'Ja', 'ja-jpn', 'ja-JA', 'ja, en', 'ja, fr'],
    'ko': ['ko', 'ko-KR', 'KO', 'Ko', 'Korean', 'kor-kr', 'KO-kr', 'ko-ko', 'ko_KR', 'Ko-Kr'],
    'zh': ['zh', 'zh-CN', 'zh-TW', 'zh-HK', 'ZH', 'Zh', 'Chi', 'zho', 'zh-hans', 'zh-hant', 'zh_CN', 'zh_tw', 'zh-zh', 'zh-US', 'zh-CHS', 'zh-Hant-', 'zh-Hans-', 'zh-rtw'],
    'ar': ['ar', 'ar-SA', 'ar-AE', 'AR', 'ara', 'ar-EG', 'ar-LB', 'ar-SA', 'ar-SY', 'ar-MA', 'ar-JO', 'ar-KW', 'ar-IQ', 'ar-LY', 'ar-QA', 'ar-AW', 'ar-FR', 'ar-US', 'ar-eg', 'ar-lb', 'ar-sa', 'ar-sy', 'ar-ma', 'ar-jo', 'ar-kw', 'ar-iq', 'ar-ly', 'ar-qa', 'ar-aw', 'ar-fr', 'ar-us', 'ar-ara', 'ar-AR', 'ar-Sa', 'ar-LB', 'ar-KW', 'ar-MA', 'ar-QA'],
    'hi': ['hi', 'hi-IN', 'HI', 'hi-HI', 'hi-in', 'hi_IN'],
    'tr': ['tr', 'tr-TR', 'TR', 'TUR-TR', 'tr-LB', 'tr-FR'],
    'nl': ['nl', 'nl-NL', 'nl-BE', 'dut', 'nl_NL', 'NL-nl', 'NL + ENG', 'nl-fr-en'],
    'pl': ['pl', 'pl-PL', 'PL', 'pol', 'pl_PL'],
    'cs': ['cs', 'cs-CZ', 'CS', 'czech', 'cs-CS'],
    'sv': ['sv', 'sv-SE', 'SV', 'Sv', 'sve', 'Svenska', 'Swedish', 'sv_SE', 'swe-SV', 'se_sv', 'se-sv'],
    'da': ['da', 'da-DK', 'DK', 'DAN', 'dan'],
    'fi': ['fi', 'fi-FI', 'fin', 'fit'],
    'no': ['no', 'no-NO', 'NO', 'Norsk', 'nor', 'nb-no', 'nb-nb', 'nn-NO', 'NO-nb'],
    'sn': ['sn'],  # Shona
    'lt': ['lt', 'lt-LT', 'lt-lt', 'LT', 'lit-LT'],  # Lithuanian
    'el': ['el', 'el-EL', 'el-el', 'el-gr', 'gre'],  # Greek
    'cmn': ['cmn'],  # Mandarin Chinese
    'my': ['my', 'my-MM', 'Burmese'],  # Burmese
    'fa': ['fa', 'fa-FA', 'fa-fa', 'fa-AF', 'fa-af', 'fa-TJ', 'fa-prd', 'fa-LB', 'Fa-Persi'],  # Persian
    'ia': ['ia'],  # Interlingua
    'ga': ['ga', 'ga-ie'],  # Irish
    'nr': ['nr'],  # South Ndebele
    'om': ['om', 'om-ET'],  # Oromo
    'ig': ['ig', 'ig-NG'],  # Igbo
    'mg': ['mg', 'mg-MG'],  # Malagasy
    'mi': ['mi'],  # Maori
    'tn': ['tn'],  # Tswana
    'ur': ['ur', 'ur-PK', 'ur-ur', 'urdu', 'Ur'],  # Urdu
    'grc': ['grc'],  # Ancient Greek
    'prs': ['prs'],  # Dari
    'ro': ['ro', 'ro-RO', 'ro_RO', 'ro-MD'],  # Romanian
    'hu': ['hu', 'hu-HU', 'HU', 'hungaria', 'hun'],  # Hungarian
    'bg': ['bg', 'bg-BG', 'BG'],  # Bulgarian
    'he': ['he', 'he-IL', 'he_IL', 'heb'],  # Hebrew
    'vi': ['vi', 'vi-VN', 'vi-vi', 'vi-VI', 'vi-US', 'vi-TH', 'vn'],  # Vietnamese
    'th': ['th', 'th-TH', 'th-Th', 'th-us', 'tha', 'TH'],  # Thai
    'id': ['id', 'id-ID', 'id-id', 'ID', 'in-in'],  # Indonesian
    'ms': ['ms', 'ms-MY', 'ms-ms', 'ms_MY'],  # Malay
    'fil': ['fil', 'fil-PH', 'fil-fil'],  # Filipino
    'uk': ['uk', 'uk-UK', 'ukr'],  # Ukrainian
    'hr': ['hr', 'hr-HR', 'hr-HRV', 'hr-hr', 'hrv', 'HR'],  # Croatian
    'sk': ['sk', 'sk-SK', 'sk–SK'],  # Slovak
    'sl': ['sl', 'sl-SI', 'sl-sl', 'sl_SI'],  # Slovenian
    'et': ['et', 'et-EE', 'et-EST', 'et-et', 'ee-ee'],  # Estonian
    'lv': ['lv', 'lv-LV', 'lv-lv', 'LV', 'latviešu'],  # Latvian
    'hy': ['hy', 'hy-AM', 'hy-hy'],  # Armenian
    'ka': ['ka', 'ka-GE'],  # Georgian
    'kk': ['kk', 'kk-KZ', 'kk-kk'],  # Kazakh
    'mn': ['mn', 'mn-MN', 'mn-mn'],  # Mongolian
    'ne': ['ne', 'ne-NP'],  # Nepali
    'si': ['si', 'si-LK'],  # Sinhala
    'ta': ['ta', 'ta-IN', 'ta-in', 'te-in', 'te-te', 'te-IN', 'tam', 'telugu'],  # Tamil/Telugu
    'bn': ['bn', 'bn-IN', 'bn-BD', 'bn-MM', 'bn-bn', 'bn-in', 'BN'],  # Bengali
    'gu': ['gu', 'gu-IN'],  # Gujarati
    'kn': ['kn', 'kn-IN', 'kn-kn', 'KN'],  # Kannada
    'ml': ['ml', 'ml-IN', 'ml-ml', 'ML', 'mal'],  # Malayalam
    'mr': ['mr', 'mr-mr', 'MR'],  # Marathi
    'pa': ['pa', 'pa-IN', 'pa-PK', 'pa-pa'],  # Punjabi
    'te': ['te', 'te-IN', 'te-in', 'te-te', 'TE'],  # Telugu
}

# Add this dictionary at the top of the file, after the imports
LANGUAGE_NAMES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh': 'Chinese',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'tr': 'Turkish',
    'nl': 'Dutch',
    'pl': 'Polish',
    'cs': 'Czech',
    'sv': 'Swedish',
    'da': 'Danish',
    'fi': 'Finnish',
    'no': 'Norwegian',
    'sn': 'Shona',
    'lt': 'Lithuanian',
    'el': 'Greek',
    'cmn': 'Mandarin Chinese',
    'my': 'Burmese',
    'fa': 'Persian',
    'ia': 'Interlingua',
    'ga': 'Irish',
    'nr': 'South Ndebele',
    'om': 'Oromo',
    'ig': 'Igbo',
    'mg': 'Malagasy',
    'mi': 'Maori',
    'tn': 'Tswana',
    'ur': 'Urdu',
    'grc': 'Ancient Greek',
    'prs': 'Dari',
    'ro': 'Romanian',
    'hu': 'Hungarian',
    'bg': 'Bulgarian',
    'he': 'Hebrew',
    'vi': 'Vietnamese',
    'th': 'Thai',
    'id': 'Indonesian',
    'ms': 'Malay',
    'fil': 'Filipino',
    'uk': 'Ukrainian',
    'hr': 'Croatian',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'et': 'Estonian',
    'lv': 'Latvian',
    'hy': 'Armenian',
    'ka': 'Georgian',
    'kk': 'Kazakh',
    'mn': 'Mongolian',
    'ne': 'Nepali',
    'si': 'Sinhala',
    'ta': 'Tamil',
    'bn': 'Bengali',
    'gu': 'Gujarati',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'mr': 'Marathi',
    'pa': 'Punjabi',
    'te': 'Telugu'
}

def clean_language_code(lang):
    """Clean and standardize language codes."""
    if not lang:
        return None
    
    # Remove any non-alphanumeric characters
    lang = re.sub(r'[^a-zA-Z0-9-]', '', lang.lower())
    
    # Check if it matches any known language code
    for main_lang, variants in LANGUAGE_MAPPING.items():
        if lang in variants:
            return main_lang
    
    return None

async def download_rss(session, url, podcast_id):
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                content = await response.text()
                
                # Generate a safe filename
                parsed_url = urlparse(url)
                filename = f"{podcast_id}_{os.path.basename(parsed_url.path)}"
                if not filename.endswith('.xml'):
                    filename += '.xml'
                
                # Save the RSS content
                filepath = os.path.join(RSS_BACKUP_DIR, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                return filepath
            else:
                print(f"Error downloading RSS for podcast {podcast_id}: HTTP {response.status}")
                return None
    except Exception as e:
        print(f"Error downloading RSS for podcast {podcast_id}: {str(e)}")
        return None

async def download_all_rss(podcasts):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for podcast in podcasts:
            podcast_id = podcast[0]
            url = podcast[4]
            tasks.append(download_rss(session, url, podcast_id))
        
        # Run all downloads concurrently with a limit of 10 concurrent requests
        results = []
        for i in range(0, len(tasks), 10):
            batch = tasks[i:i+10]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)
            # Small delay between batches
            await asyncio.sleep(0.5)
        
        return results

def main():
    conn = sqlite3.connect("podcastindex_feeds.db")
    cursor = conn.cursor()

    # Calculate date 1 year ago
    one_year_ago = datetime.now() - timedelta(days=360)
    one_year_ago_timestamp = int(one_year_ago.timestamp())

    # Get recent podcasts from all languages
    cursor.execute("""
        SELECT id, title, language, description, url, lastupdate
        FROM podcasts 
        WHERE lastupdate >= ?
        ORDER BY lastupdate DESC
    """, (one_year_ago_timestamp,))

    # Fetch all results
    all_podcasts = cursor.fetchall()

    # First, analyze language distribution
    raw_language_counts = {}
    for podcast in all_podcasts:
        lang = podcast[2]
        if lang:
            raw_language_counts[lang] = raw_language_counts.get(lang, 0) + 1

    print("\nRaw language distribution in the database:")
    print("Language code -> Number of podcasts")
    print("-" * 40)
    for lang, count in sorted(raw_language_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{lang:15} -> {count:5} podcasts")

    # Now process and group podcasts by cleaned language codes
    language_podcasts = {}
    for podcast in all_podcasts:
        lang = clean_language_code(podcast[2])
        if lang:
            if lang not in language_podcasts:
                language_podcasts[lang] = []
            language_podcasts[lang].append(podcast)

    # Filter languages with at least 5 podcasts
    valid_languages = {lang: podcasts for lang, podcasts in language_podcasts.items() 
                      if len(podcasts) >= 5}

    print("\nCleaned language distribution (languages with ≥5 podcasts):")
    print("Language code -> Number of podcasts")
    print("-" * 40)
    for lang, podcasts in sorted(valid_languages.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"{lang:15} -> {len(podcasts):5} podcasts")

    # Create a dictionary to store the RSS addresses
    rss_data = {
        "total_languages": len(valid_languages),
        "languages": {}
    }

    # For each language, get the 10 most recent podcasts
    for lang, podcasts in valid_languages.items():
        # Sort by lastupdate and take up to 100 (or fewer if not enough)
        sorted_podcasts = sorted(podcasts, key=lambda x: x[5], reverse=True)
        top_podcasts = sorted_podcasts[:100] if len(sorted_podcasts) > 100 else sorted_podcasts

        rss_data["languages"][lang] = {
            "language_name": LANGUAGE_NAMES.get(lang, "Unknown"),
            "podcasts": [
                {
                    "id": podcast[0],
                    "title": podcast[1],
                    "language_code": lang,
                    "language_name": LANGUAGE_NAMES.get(lang, "Unknown"),
                    "description": podcast[3],
                    "url": podcast[4],
                    "last_update": datetime.fromtimestamp(podcast[5]).strftime('%Y-%m-%d %H:%M:%S')
                }
                for podcast in top_podcasts
            ]
        }

    # Save to JSON file
    with open('multilingual_podcasts_rss.json', 'w', encoding='utf-8') as f:
        json.dump(rss_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved RSS addresses for {len(valid_languages)} languages to multilingual_podcasts_rss.json")
    print(f"Total number of podcasts: {sum(len(data['podcasts']) for data in rss_data['languages'].values())}")

    conn.close()

if __name__ == "__main__":
    main() 