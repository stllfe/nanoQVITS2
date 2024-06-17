"""Quantized features computation utilities."""

from __future__ import annotations

import copy

from dataclasses import dataclass
from typing import Iterable, NamedTuple

import numpy as np

from numpy.typing import NDArray
from textgrid import TextGrid

from utils.audio import compute_pitch
from utils.audio import compute_pitch_slope
from utils.helpers import debug


PAD_FEATURE = 0


@dataclass
class Word:
    text: str
    index: int
    start: float
    end: float
    feats: WordFeatures


class WordFeatures(NamedTuple):
    volume: float
    speed: float
    pitch_mean: float
    pitch_fslope: float
    pitch_lslope: float
    pitch_rslope: float

    def __array__(self) -> NDArray:
        return np.fromiter(self, dtype=float)


def isspecial(word: str) -> bool:
    """Checks wether the given word is in format `<word>`."""

    return word.startswith('<') and word.endswith('>')


def compute_word_features(
    audio: NDArray,
    alignment: TextGrid,
    /,
    rate: int = 22050,
    min_duration: float = 0.01,
) -> Iterable[Word]:
    """Computes word-level features from the alignment."""

    assert len(alignment) > 0, 'Empty alignment!'
    times, f0 = compute_pitch(audio, rate)

    markup = alignment[0]
    for index, interval in enumerate(markup):
        t0, t1, word = interval.minTime, interval.maxTime, interval.mark
        dt = t1 - t0

        if not word or isspecial(word) or dt < min_duration:
            continue

        f0w = f0[(times >= t0) & (times <= t1)]
        chunk = audio[int(rate * t0) : int(rate * t1)]
        mid = len(f0w) // 2

        yield Word(
            text=word,
            index=index,
            start=t0,
            end=t1,
            feats=WordFeatures(
                volume=np.std(chunk).item(),
                speed=len(word) / dt,
                pitch_mean=np.mean(f0w).item(),
                pitch_fslope=compute_pitch_slope(f0w),
                pitch_lslope=compute_pitch_slope(f0w[:mid]),
                pitch_rslope=compute_pitch_slope(f0w[mid:]),
            ),
        )


def quantize_features(words: Iterable[Word], bins: int = 5) -> Iterable[Word]:
    """Quantizes word-level features to the specified number of bins."""

    feats = np.array([w.feats for w in words])
    for j in range(feats.shape[1]):
        debug(f'Feature: {j}', level=2)
        # todo: mean imputation maybe not the best idea though
        mean = np.nanmean(feats[:, j]).item()
        debug(f'Filling NaNs with {mean=:.4f}', level=2)
        np.nan_to_num(feats[:, j], nan=mean, copy=False)
        _, edges = np.histogram(feats[:, j], bins=bins)
        debug(f'Edges: {edges}\n', level=2)
        feats[:, j] = np.digitize(feats[:, j], bins=edges, right=True)
    feats = feats.astype(np.uint8)
    for i, w in enumerate(words):
        w = copy.deepcopy(w)
        w.feats = WordFeatures(*feats[i])
        yield w


def compute_word_spans(text: str, words: Iterable[Word]) -> NDArray:
    """Computes word spans [start, end] in the given text."""

    words = iter(words)
    word = next(words, None)
    assert text and word, 'Both sequences should be non-empty!'

    i = 0
    spans = []
    while i < len(text):
        if word and text[i : i + len(word.text)].startswith(word.text):
            spans.append([i, i + len(word.text)])
            i += len(word.text)
            word = next(words, None)
        else:
            i += 1
    assert len(list(words)) == 0, 'Not all words exist in the given text!'
    return np.array(spans, dtype=np.uint32)


def printchr(text: str, words: Iterable[Word], spans: NDArray | None = None) -> None:
    """Prints the word-level features aligned to char text."""

    spans = compute_word_spans(text, words) if spans is None else spans
    assert len(spans), 'Word spans should be non-empty!'

    word = next(iter(words))
    assert word, 'Empty words sequence!'

    features = np.full((len(text), len(word.feats)), fill_value=PAD_FEATURE, dtype=np.uint8)
    for word, span in zip(words, spans):
        s, e = span
        features[s:e, :] = np.array(word.feats)

    print(text)
    for line in features.T:
        print(''.join(map(str, line)))
