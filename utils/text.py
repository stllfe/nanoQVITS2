import re

from unidecode import unidecode


_WHITESPACE_PATTERN = re.compile(r'\s+')
_ABBREVIATIONS = [
    (re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1])
    for x in [
        ('mrs', 'misess'),
        ('mr', 'mister'),
        ('dr', 'doctor'),
        ('st', 'saint'),
        ('co', 'company'),
        ('jr', 'junior'),
        ('maj', 'major'),
        ('gen', 'general'),
        ('drs', 'doctors'),
        ('rev', 'reverend'),
        ('lt', 'lieutenant'),
        ('hon', 'honorable'),
        ('sgt', 'sergeant'),
        ('capt', 'captain'),
        ('esq', 'esquire'),
        ('ltd', 'limited'),
        ('col', 'colonel'),
        ('ft', 'fort'),
    ]
]

_PAD = '_'
_PUNCTUATION = ';:,.!?—…"«»“” '
_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

SYMBOLS = [_PAD] + list(_PUNCTUATION) + list(_LETTERS)
PAD_IDX = SYMBOLS.index(_PAD)

SYMBOL_TO_INT = {s: i for i, s in enumerate(SYMBOLS)}
INT_TO_SYMBOL = {i: s for i, s in enumerate(SYMBOLS)}


def expand_abbreviations(text: str) -> str:
    for pattern, repl in _ABBREVIATIONS:
        text = re.sub(pattern, repl, text)
    return text


def collapse_whitespace(text: str) -> str:
    return re.sub(_WHITESPACE_PATTERN, ' ', text)


def convert_to_ascii(text: str) -> str:
    return unidecode(text)


def clean(text: str) -> str:
    text = convert_to_ascii(text)
    text = expand_abbreviations(text.lower())
    text = collapse_whitespace(text)
    return text


def encode_text(text: str) -> list[int]:
    """Converts a string of text to a sequence of IDs corresponding to the symbols."""

    return [SYMBOL_TO_INT[s] for s in text if s in SYMBOL_TO_INT]


def decode_text(sequence: list[int]) -> str:
    """Converts a sequence of IDs back to a string."""

    return ''.join([INT_TO_SYMBOL[i] for i in sequence])
