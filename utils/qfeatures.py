"""q-Features computation utilities."""

from __future__ import annotations

import copy

from dataclasses import dataclass
from typing import Iterable, NamedTuple

import numpy as np

from numpy.typing import NDArray
from textgrid import TextGrid

from utils.audio import compute_pitch, compute_pitch_slope


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

    for index, interval in enumerate(alignment[0]):
        t0, t1, word = interval.minTime, interval.maxTime, interval.mark
        dt = t1 - t0

        if dt < min_duration or not word:
            continue

        f0w = f0[(times >= t0) & (times <= t1)]
        chunk = audio[int(rate * t0):int(rate * t1)]
        m = len(f0w) // 2

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
                pitch_lslope=compute_pitch_slope(f0w[:m]),
                pitch_rslope=compute_pitch_slope(f0w[m:]),
            )
        )


def quantize_features(words: Iterable[Word], bins: int = 5) -> Iterable[Word]:
    """Quantizes word-level features to the specified number of bins."""

    feats = np.array([w.feats for w in words])
    for j in range(feats.shape[1]):
        print(f'Feature: {j}')
        # todo: mean imputation maybe not the best idea though
        mean = np.nanmean(feats[:, j])
        print(f'Filling NaNs with {mean=:.4f}')
        np.nan_to_num(feats[:, j], nan=mean, copy=False)
        _, edges = np.histogram(feats[:, j], bins=bins)
        print(f'Edges: {edges}\n')
        feats[:, j] = np.digitize(feats[:, j], bins=edges, right=False)
    # shift values by 1 -> [0, bins)
    feats = feats.astype(np.uint8) - 1
    for i, w in enumerate(words):
        w = copy.deepcopy(w)
        w.feats = WordFeatures(*feats[i])
        yield w
