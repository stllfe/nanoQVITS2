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


def compute_word_spans(text: str, words: Iterable[Word]) -> NDArray:
    """Computes word spans [start, end] in the given text."""

    words = iter(words)
    word = next(words, None)
    assert text and word, 'Both sequences should be non-empty!'

    i = 0
    spans = []
    while i < len(text):
        if word and text[i: i + len(word.text)].startswith(word.text):
            spans.append([i, i + len(word.text)])
            i += len(word.text)
            word = next(words, None)
        else:
            i += 1
    return np.array(spans)


def printchr(text: str, words: Iterable[Word], spans: NDArray | None = None) -> None:
    """Prints the word-level features aligned to char text."""

    spans = compute_word_spans(text, words) if spans is None else spans
    assert len(spans), 'Word spans should be non-empty!'

    word = next(iter(words))
    assert word

    features = np.full((len(text), len(word.feats)), fill_value=PAD_FEATURE, dtype=np.uint8)
    for word, span in zip(words, spans):
        s, e = span
        features[s:e,:] = np.array(word.feats)

    print(text)
    for line in features.T:
        print(''.join(map(str, line)))


# if __name__ == '__main__':
#     text = 'ah, shit! here we go again...'
#     words = [
#         Word(text='ah', index=0, start=0, end=1, feats=WordFeatures(volume=1, speed=1, pitch_mean=1, pitch_fslope=1, pitch_lslope=1, pitch_rslope=2)),
#         Word(text='shit', index=1, start=0, end=1, feats=WordFeatures(volume=2, speed=2, pitch_mean=2, pitch_fslope=2, pitch_lslope=2, pitch_rslope=3)),
#         Word(text='here', index=1, start=0, end=1, feats=WordFeatures(volume=3, speed=3, pitch_mean=3, pitch_fslope=3, pitch_lslope=3, pitch_rslope=4)),
#         Word(text='we', index=1, start=0, end=1, feats=WordFeatures(volume=4, speed=4, pitch_mean=4, pitch_fslope=4, pitch_lslope=4, pitch_rslope=5)),
#         Word(text='go', index=1, start=0, end=1, feats=WordFeatures(volume=5, speed=5, pitch_mean=5, pitch_fslope=5, pitch_lslope=5, pitch_rslope=6))
#     ]
#     print(compute_word_spans(text, words))
#     printchr(text, words)
