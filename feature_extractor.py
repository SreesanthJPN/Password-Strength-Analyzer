import re
import math
import string
import nltk
import spacy
from collections import Counter
from nltk.corpus import words, cmudict

# Download necessary resources
nltk.download('words', quiet=True)
nltk.download('cmudict', quiet=True)

# Load NLP tools
english_words = set(words.words())
syllable_dict = cmudict.dict()
nlp = spacy.load("en_core_web_sm")


# --- Helper Functions ---

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    probabilities = [n / len(s) for n in Counter(s).values()]
    return -sum(p * math.log2(p) for p in probabilities)


def count_syllables(word: str) -> int:
    word = word.lower()
    if word in syllable_dict:
        return max(len([y for y in x if y[-1].isdigit()]) for x in syllable_dict[word])
    else:
        return len(re.findall(r'[aeiouy]+', word.lower()))


def is_english_word(word: str) -> bool:
    return word.lower() in english_words


def is_palindrome(word: str) -> bool:
    return word.lower() == word[::-1].lower()


def char_repeat_ratio(word: str) -> float:
    if not word:
        return 0.0
    counts = Counter(word.lower())
    most_common = counts.most_common(1)[0][1]
    return most_common / len(word)


def get_keyboard_patterns():
    return [
        # QWERTY rows
        "qwertyuiop", "asdfghjkl", "zxcvbnm",
        "poiuytrewq", "lkjhgfdsa", "mnbvcxz",
        # Number rows
        "1234567890", "0987654321",
        # Symbols
        "!@#$%^&*()", ")(*&^%$#@!", "`~!@#$%^&*()_+-=", "-=_+)(*&^%$#@!~`",
        # Brackets/punctuations
        "[]{}", "{}[]", "<>", "><", "()", ")(",
        # Diagonals
        "qaz", "wsx", "edc", "rfv", "tgb", "yhn", "ujm",
        "zaq", "xsw", "cde", "vfr", "bgt", "nhy", "mju",
        # Numpad patterns
        "147", "258", "369", "741", "852", "963", "159", "357"
    ]


def has_keyboard_pattern(password: str, patterns=None, min_len: int = 3) -> bool:
    if patterns is None:
        patterns = get_keyboard_patterns()
    pw = password.lower()
    for pattern in patterns:
        for i in range(len(pattern) - min_len + 1):
            sub = pattern[i:i+min_len]
            if sub in pw:
                return True
    return False


def contains_english_word_substring(password: str, min_len=3) -> bool:
    pw = password.lower()
    for i in range(len(pw)):
        for j in range(i + min_len, len(pw) + 1):
            if pw[i:j] in english_words:
                return True
    return False


def contains_ner_substring(password: str, min_len=3) -> bool:
    for i in range(len(password)):
        for j in range(i + min_len, len(password) + 1):
            sub = password[i:j]
            doc = nlp(sub)
            if any(ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC'] for ent in doc.ents):
                return True
    return False


# --- Main Feature Extractor ---

def extract_password_features(password: str) -> dict:
    nlp_doc = nlp(password)
    alphabet_count = sum(c.isalpha() for c in password)
    digit_count = sum(c.isdigit() for c in password)
    special_count = sum(c in string.punctuation for c in password)
    uppercase_count = sum(c.isupper() for c in password)
    entropy = shannon_entropy(password)
    syllables = count_syllables(password)
    is_word = is_english_word(password)
    palindrome = is_palindrome(password)
    repeat_ratio = char_repeat_ratio(password)
    ner_count = sum(1 for ent in nlp_doc.ents)
    keyboard_pattern = has_keyboard_pattern(password)
    has_ner_substr = contains_ner_substring(password)
    has_english_word_substr = contains_english_word_substring(password)

    return {
        'alphabet_count': alphabet_count,
        'digit_count': digit_count,
        'special_count': special_count,
        'uppercase_count': uppercase_count,
        'entropy': entropy,
        'syllables': syllables,
        'is_english_word': int(is_word),
        'is_palindrome': int(palindrome),
        'repeat_char_ratio': repeat_ratio,
        'ner_count': ner_count,
        'has_keyboard_pattern': int(keyboard_pattern),
        'contains_ner_substring': int(has_ner_substr),
        'contains_english_word_substring': int(has_english_word_substr)
    }
