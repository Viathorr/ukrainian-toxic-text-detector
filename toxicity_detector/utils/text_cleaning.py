import re
import unicodedata
import emoji

USERNAME_REGEX = r"@\w+"
HASHTAG_REGEX = r"#\w+"
TAGS_REGEX = r"<[^>]+>"
REPEATED_CHAR_REGEX = r"(.)\1{2,}"  # Matches any character repeated more than twice
REPEATED_PUNCTUATION_REGEX = r"([^\w\s])\1+"  # Matches punctuation repeated more than once
URL_REGEX = r"https?:\/\/\S+|www\.\S+"
WHITESPACE_REGEX = r"\s+"  # Matches any whitespace character (spaces, tabs, newlines)

# Wiki elements regex patterns:
WIKI_CLEAN_REGEX = [
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",  # IP addresses
    r"\b\d{1,2}:\d{2},\s+[A-Z][a-z]+\s+\d{1,2},\s+\d{4}\s*\(UTC\)",  # timestamps
    r"\{\{.*?\}\}",                                               # {{templates}}
    r"\[\[.*?\]\]",                                               # [[links/files]]
    r"^:+"                                                       # thread indentation (::::)
]

# TODO: Add email pattern for removing emails

def remove_wiki_elements(text: str) -> str:
    """
    Remove wiki-specific elements from the text.
    Intended for pre-translation cleaning only.
    """
    for pattern in WIKI_CLEAN_REGEX:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE | re.MULTILINE)
    return text

def handle_mentions(text: str) -> str:
    """Replace @username with [USER]."""
    return re.sub(USERNAME_REGEX, "[USER]", text)

def handle_hashtags(text: str) -> str:
    """Replace #hashtag with hashtag."""
    return re.sub(HASHTAG_REGEX, lambda x: x.group()[1:], text)

def remove_tags(text: str) -> str:
    """Remove HTML/XML tags from the text."""
    return re.sub(TAGS_REGEX, "", text)

def reduce_repetitions(text: str) -> str:
    """Reduce characters repeated more than twice to two occurrences."""
    return re.sub(REPEATED_CHAR_REGEX, lambda x: x.group(1) * 2, text)

def remove_repetitive_punctuation(text: str) -> str:
    """Remove punctuation repeated more than once."""
    return re.sub(REPEATED_PUNCTUATION_REGEX, lambda x: x.group(1), text)

def remove_repetitive_chunks(text: str, max_repeats: int = 2) -> str:
    """Detects and collapses consecutive repeating word sequences."""
    if not text:
        return ""

    pattern = re.compile(r"(.+?)(?:\s+\1){" + str(max_repeats) + r",}", re.IGNORECASE)
    
    for _ in range(2):
        text = pattern.sub(r"\1", text)
        
    return text

def remove_repeated_sentences(text: str) -> str:
    """Remove repeated sentences within a single comment."""
    # Split into sentences (roughly)
    sentences = re.split(r'(?<=[.!?]) +', text)
    seen = set()
    unique_sentences = []
    
    for s in sentences:
        if s.lower() not in seen: 
            seen.add(s.lower())
            unique_sentences.append(s)
            
    return ' '.join(unique_sentences)

def remove_urls(text: str) -> str:
    """Replace URLs with [URL]."""
    return re.sub(URL_REGEX, "[URL]", text, flags=re.MULTILINE)

def convert_emojis_to_text(text: str) -> str:
    """Convert emojis to their text representation."""
    return emoji.demojize(text, delimiters=(":", ":"))

def remove_newlines_and_whitespace(text: str) -> str:
    """Remove newlines and extra whitespace from the text."""
    return re.sub(WHITESPACE_REGEX, ' ', text).strip()

def remove_nonprintable(text: str) -> str:
    return ''.join(c for c in text if c.isprintable())

def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters to NFC form."""
    return unicodedata.normalize("NFC", text)