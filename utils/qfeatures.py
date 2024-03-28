"""Quantized features computation utilities."""

from __future__ import annotations

import copy

from dataclasses import dataclass
from typing import Iterable, NamedTuple

import numpy as np

from numpy.typing import NDArray
from textgrid import TextGrid

from utils.audio import compute_pitch, compute_pitch_slope
from utils.helpers import DEBUG


UNK_FEATURE = 0


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
    prev = 0
    for index, interval in enumerate(markup):
        t0, t1, word = interval.minTime, interval.maxTime, interval.mark
        dt = t1 - t0

        if dt < min_duration or not word:
            continue

        f0w = f0[(times >= t0) & (times <= t1)]
        chunk = audio[int(rate * t0):int(rate * t1)]
        mid = len(f0w) // 2

        yield Word(
            text=word,
            index=index,
            start=t0,
            end=t1,
            feats=WordFeatures(
                volume=np.std(chunk).item(),
                speed=len(word) / dt if not isspecial(word) else np.nan,
                pitch_mean=np.mean(f0w).item(),
                pitch_fslope=compute_pitch_slope(f0w),
                pitch_lslope=compute_pitch_slope(f0w[:mid]),
                pitch_rslope=compute_pitch_slope(f0w[mid:]),
            )
        )


def quantize_features(words: Iterable[Word], bins: int = 5) -> Iterable[Word]:
    """Quantizes word-level features to the specified number of bins."""

    feats = np.array([w.feats for w in words])
    for j in range(feats.shape[1]):
        if DEBUG: print(f'Feature: {j}')
        # todo: mean imputation maybe not the best idea though
        mean = np.nanmean(feats[:, j]).item()
        if DEBUG: print(f'Filling NaNs with {mean=:.4f}')
        np.nan_to_num(feats[:, j], nan=mean, copy=False)
        _, edges = np.histogram(feats[:, j], bins=bins)
        if DEBUG: print(f'Edges: {edges}\n')
        feats[:, j] = np.digitize(feats[:, j], bins=edges, right=True)
    feats = feats.astype(np.uint8)
    for i, w in enumerate(words):
        w = copy.deepcopy(w)
        w.feats = WordFeatures(*feats[i])
        yield w


def getchr(text: str, words: Iterable[Word]) -> NDArray:
    """Populates word-level q-features per each text character as an NDArray."""

    words = iter(words)
    word = next(words, None)
    assert text and word, 'Both sequences should be non-empty!'

    pads = [UNK_FEATURE] * len(word.feats)
    i = 0
    features = []
    while i < len(text):
        char = text[i]
        if word and word.text.startswith(char):
            features.extend([list(word.feats)]*len(word.text))
            i += len(word.text)
            word = next(words, None)
        else:
            features.append(pads)
            i += 1
    return np.array(features)


def printchr(text: str, words: Iterable[Word]) -> None:
    """Prints the word-level features aligned to char text."""

    print(text)
    for line in getchr(text, words).T:
        print(''.join(map(str, line)))


def gettok(text: str, words: Iterable[Word], offsets: NDArray) -> NDArray:
    """Populates word-level q-features per each token as an NDArray."""

    assert len(offsets), 'Offsets should be non-empty!'
    features = []
    chrs = getchr(text, words)
    for offset in offsets:
        s, e = offset
        tok = text[s:e]
        print(tok)
        assert np.equal(chrs[s:e, ...], chrs[s]).all(), "Characters and tokens are not aligned!"
        features.append(chrs[s])
    return np.array(features)
