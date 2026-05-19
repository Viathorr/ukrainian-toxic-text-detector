import re
import unicodedata
import emoji

USERNAME_REGEX = r"(?<!\w)@\w+"
HASHTAG_REGEX = r"#\w+"
TAGS_REGEX = r"<[^>]+>"
REPEATED_CHAR_REGEX = r"(.)\1{2,}"  # Matches any character repeated more than twice
URL_REGEX = r"https?:\/\/\S+|www\.\S+"
EMAIL_REGEX = r"\S+@\S+\.\S+"
PHONE_NUMBER_REGEX = r"(\+?\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}"
WHITESPACE_REGEX = r"\s+"  # Matches any whitespace character (spaces, tabs, newlines)

# Wiki elements regex patterns:
WIKI_CLEAN_REGEX = [
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",  # IP addresses
    r"\b\d{1,2}:\d{2},\s+[A-Z][a-z]+\s+\d{1,2},\s+\d{4}\s*\(UTC\)",  # timestamps
    r"\{\{.*?\}\}",                                               # {{templates}}
    r"\[\[.*?\]\]",                                               # [[links/files]]
    r"^:+"                                                       # thread indentation (::::)
]


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

def remove_hashtags(text: str) -> str:
    """Remove hashtags."""
    return re.sub(HASHTAG_REGEX, "", text)

def remove_tags(text: str) -> str:
    """Remove HTML/XML tags from the text."""
    return re.sub(TAGS_REGEX, "", text)

def reduce_repetitions(text: str) -> str:
    """Reduce characters repeated more than twice to two occurrences."""
    return re.sub(REPEATED_CHAR_REGEX, lambda x: x.group(1) * 2, text)

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

def handle_urls_emails(text: str) -> str:
    """Replace URLs and emails with [URL] and [EMAIL]."""
    text = re.sub(URL_REGEX, "[URL]", text, flags=re.MULTILINE)
    text = re.sub(EMAIL_REGEX, "[EMAIL]", text, flags=re.MULTILINE)
    return text

def handle_phone_numbers(text: str) -> str:
    """Replace phone numbers with [PHONE]."""
    return re.sub(PHONE_NUMBER_REGEX, "[PHONE]", text)

def convert_emojis_to_text(text: str) -> str:
    """Convert emojis to their text representation."""
    return emoji.demojize(text, delimiters=(":", ":"))

def remove_newlines_and_whitespace(text: str) -> str:
    """Remove newlines and extra whitespace from the text."""
    return re.sub(WHITESPACE_REGEX, ' ', text).strip()

def remove_nonprintable(text: str) -> str:
    """Remove non-printable characters from the text."""
    return ''.join(c for c in text if c.isprintable())

def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters to NFC form."""
    return unicodedata.normalize("NFC", text)


class TextPreprocessor:
    """
    Preprocessor for text data.
    
    IMPORTANT:
    - This preprocessor is intended for training and inference normalization.
    - Heavy dataset-cleaning steps (e.g. aggressive deduplication, wiki stripping)
    should be explicitly enabled via flags.
    
    Examples
    --------
    >>> preprocessor = TextPreprocessor()
    >>> text = "@user Check this out https://example.com 😊"
    >>> preprocessor.clean_text(text)
    '[USER] Check this out [URL] :smiling_face_with_smiling_eyes:'
    
    >>> # Custom configuration
    >>> preprocessor = TextPreprocessor(lowercase=True, remove_hashtags=True)
    >>> preprocessor.clean_text("Python is awesome! #python #programming")
    'python is awesome! '
    """
    
    def __init__(
        self,
        remove_urls_and_emails: bool = True,
        remove_tags: bool = True,
        remove_mentions: bool = True,
        remove_hashtags: bool = True,
        remove_phone_numbers: bool = True,
        convert_emojis: bool = True,
        lowercase: bool = False,
        remove_newlines_and_spaces: bool = True,
        remove_nonprintable_characters: bool = False,
        remove_repeated_sentences: bool = False,
        remove_repeated_chunks: bool = False,
        reduce_repetitions: bool = True,
        normalize_unicode: bool = True,
        remove_wiki_elements: bool = False
        ):
        self.config = {
            "remove_urls_and_emails": remove_urls_and_emails,
            "remove_tags": remove_tags,
            "remove_mentions": remove_mentions,
            "remove_hashtags": remove_hashtags,
            "remove_phone_numbers": remove_phone_numbers,
            "convert_emojis": convert_emojis,
            "lowercase": lowercase,
            "remove_newlines_and_spaces": remove_newlines_and_spaces,
            "remove_nonprintable_characters": remove_nonprintable_characters,
            "remove_repeated_sentences": remove_repeated_sentences,
            "remove_repeated_chunks": remove_repeated_chunks,
            "reduce_repetitions": reduce_repetitions,
            "normalize_unicode": normalize_unicode,
            "remove_wiki_elements": remove_wiki_elements
        }
        
    def clean_text(self, text: str) -> str:
        """
        Clean a given text based on the preprocessor configuration.
        
        Applies various text cleaning operations such as removing URLs and emails, tags, mentions, hashtags, phone numbers, 
        converting emojis to their text representation, and more.
        
        Parameters
        ----------
        text : str
            Text to be cleaned.
        
        Returns
        -------
        str
            Cleaned text.
        """
    
        if self.config["remove_wiki_elements"]:
            text = remove_wiki_elements(text)
        if self.config["remove_tags"]:
            text = remove_tags(text)
        if self.config["remove_hashtags"]:
            text = remove_hashtags(text)
        if self.config["remove_urls_and_emails"]:
            text = handle_urls_emails(text)
        if self.config["remove_phone_numbers"]:
            text = handle_phone_numbers(text)
        if self.config["remove_mentions"]:
            text = handle_mentions(text)
        if self.config["convert_emojis"]:
            text = convert_emojis_to_text(text)
        if self.config["lowercase"]:
            text = text.lower()
        if self.config["remove_repeated_chunks"]:
            text = remove_repetitive_chunks(text)
        if self.config["remove_repeated_sentences"]:
            text = remove_repeated_sentences(text)
        if self.config["reduce_repetitions"]:
            text = reduce_repetitions(text)
        if self.config["normalize_unicode"]:
            text = normalize_unicode(text)
        if self.config["remove_nonprintable_characters"]:
            text = remove_nonprintable(text)
        if self.config["remove_newlines_and_spaces"]:
            text = remove_newlines_and_whitespace(text)
            
        return text.strip()