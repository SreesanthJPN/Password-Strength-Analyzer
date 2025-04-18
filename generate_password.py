import random
import string
import nltk
from nltk.corpus import words, names
from feature_extractor import extract_password_features

nltk.download('words')
nltk.download('names')

english_words = set(words.words())
ner_words = set(names.words())

special_chars = "!@#$%^&*()-_=+[]{}|;:,.<>?/\\"

keyboard_patterns = [
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

def generate_strong_password(length=16):
    while True:
        # Ensure at least one from each required category
        letters = random.choices(string.ascii_lowercase, k=5)
        digits = random.choices(string.digits, k=3)
        specials = random.choices(special_chars, k=3)
        uppers = random.choices(string.ascii_uppercase, k=3)

        base = letters + digits + specials + uppers
        random.shuffle(base)

        # Add a known English word and NER word to boost features
        word = random.choice(list(english_words)).lower()
        ner_word = random.choice(list(ner_words)).capitalize()

        password = ''.join(base) + word + ner_word
        password = ''.join(random.sample(password, len(password)))  # Shuffle again

        # Validate using your existing feature extractor
        features = extract_password_features(password)


        if (
            features['alphabet_count'] >= 8 and
            features['digit_count'] >= 2 and
            features['special_count'] >= 2 and
            features['uppercase_count'] >= 2 and
            features['entropy'] > 3.5 and
            features['syllables'] >= 3 and
            features['is_english_word'] == 0 and  # whole password shouldn't be just one word
            features['is_palindrome'] == 0 and
            features['repeat_char_ratio'] < 0.3 and
            features['ner_count'] > 0 and
            features['has_keyboard_pattern'] == 0 and
            features['contains_ner_substring'] == 1 and
            features['contains_english_word_substring'] == 1
        ):
            return password
