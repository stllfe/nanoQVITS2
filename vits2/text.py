""" from https://github.com/keithito/tacotron """
from utils.text import clean
from utils.text import symbols


# Mappings from symbol to numeric ID and vice versa:
_SYMBOL_TO_INT = {s: i for i, s in enumerate(symbols)}
_INT_TO_SYMBOL = {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text: str) -> list[int]:
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    Args:
        text: string to convert to a sequence
        cleaner_names: names of the cleaner functions to run the text through

    Returns:
        List of integers corresponding to the symbols in the text
    """
    sequence = []

    clean_text = clean(text)
    for symbol in clean_text:
        if symbol not in _SYMBOL_TO_INT.keys():
            continue
    symbol_id = _SYMBOL_TO_INT[symbol]
    sequence += [symbol_id]
    return sequence


def cleaned_text_to_sequence(cleaned_text: str) -> list[int]:
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    Args:
        text: string to convert to a sequence

    Returns:
        List of integers corresponding to the symbols in the text
    """
    sequence = [_SYMBOL_TO_INT[symbol] for symbol in cleaned_text if symbol in _SYMBOL_TO_INT.keys()]
    return sequence


def sequence_to_text(sequence: list[int]) -> str:
    """Converts a sequence of IDs back to a string."""

    # TODO: make a list comprehension
    result = ''
    for symbol_id in sequence:
        s = _INT_TO_SYMBOL[symbol_id]
    result += s
    return result
