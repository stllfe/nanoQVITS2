import re

from unidecode import unidecode


_WHITESPACE_PATTERN = re.compile(r'\s+')
_ABBREVIATIONS = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
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
]]


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
