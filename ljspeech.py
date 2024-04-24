"""Download and preprocess LJSpeech dataset for qVITS2 training."""

from __future__ import annotations

from collections import defaultdict

import os
import sys

import multiprocessing as mp

from pathlib import Path
from typing import Iterable, NamedTuple, Union

sys.path.append(os.path.dirname(__file__))

import h5py
import numpy as np
import requests
import tyro

from numpy.typing import NDArray
from textgrid import TextGrid
from tqdm import tqdm

from utils import bert

from utils.audio import (
    normalize,
    readwav,
    writewav,
)
from utils.data import Utterance
from utils.text import clean, encode_text

from utils.qfeatures import (
    compute_word_features,
    compute_word_spans,
    quantize_features,
    Word,
)


NUM_WORKERS = (os.cpu_count() or 1) // 2

PROJ_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJ_DIR, 'data')

URL = 'https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2'
DIR = 'LJSpeech-1.1'

WAVS_DIR = os.path.join(DATA_DIR, DIR, 'wavs')
MFAS_DIR = os.path.join(DATA_DIR, DIR, 'alignment')
FEAT_DIR = os.path.join(DATA_DIR, DIR, 'features')


def download_file(url: str, fname: str, chunk_size: int = 1024) -> None:
    """Downloads a file from a given url."""

    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(fname, mode='wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def load_from_metadata(filename: str = 'metadata.csv') -> Iterable[Utterance]:
    """Loads dataset samples lazily."""

    filepath = os.path.join(DATA_DIR, DIR, filename)
    with open(filepath, mode='r', encoding='utf-8') as file:
        for line in file:
            name, _, text = line.split('|')
            yield Utterance(name, text.strip())


def load_from_prepared() -> Iterable[Utterance]:
    """Loads dataset samples from disk."""

    wavs_dir = Path(DATA_DIR, DIR, 'wavs')
    for p in wavs_dir.glob('*.lab'):
        with open(p, mode='r', encoding='utf-8') as file:
            text = file.readline()
            yield Utterance(p.stem, text.strip())


def download() -> None:
    """Downloads the LJSpeech dataset to data directory."""

    os.makedirs(DATA_DIR, exist_ok=True)

    # download the dataset unless it already exists
    filename = os.path.join(DATA_DIR, os.path.basename(URL))
    if not os.path.exists(filename):
        print(f'Downloading {URL} to {filename}...')
        download_file(URL, filename)
    else:
        print(f'{filename} already exists, skipping download...')

    # unpack the tar.bz2 file into ljspeech directory
    data_dir = os.path.join(DATA_DIR, DIR)

    # TODO: can we skip it and do the prepare step from the compressed tar,
    # just by sequentially reading it?
    if not os.path.exists(data_dir):
        print(f'Unpacking {filename}...')
        if os.system(f'tar -xjf {filename} -C {DATA_DIR}'):
            raise IOError(f'Error while extracting data!')
    else:
        print(f'{data_dir} already exists, skipping unpacking...')

    print('Download complete.')


def prepare(dst: str | None = None) -> None:
    """Prepares all the necessary files for alignment."""

    wavs_dir = os.path.join(DATA_DIR, DIR, 'wavs')

    for sample in tqdm(load_from_metadata(), desc='Preparing samples'):
        wavpath = os.path.join(wavs_dir, f'{sample.filename}.wav')
        try:
            wav, sr = readwav(wavpath)
        except FileNotFoundError:
            tqdm.write(f'Not found: {wavpath}', sys.stderr)
            continue
        writewav(normalize(wav), sr, os.path.join(wavs_dir, os.path.basename(wavpath)))

        labpath = os.path.join(wavs_dir, f'{sample.filename}.lab')
        with open(labpath, mode='w', encoding='utf-8') as file:
            file.write(clean(sample.text) + '\n')


def process_utterance(uttr: Utterance) -> tuple[Utterance, list[Word]]:
    wavpath = Path(WAVS_DIR, uttr.filename).with_suffix('.wav')
    mfapath = Path(MFAS_DIR, uttr.filename).with_suffix('.TextGrid')

    audio, rate = readwav(wavpath.as_posix())
    alignment = TextGrid.fromFile(mfapath.as_posix())
    words = compute_word_features(audio, alignment, rate=rate, min_duration=0.01)

    return uttr, list(words)


def process() -> None:
    words: list[Word] = []
    uttrs: list[Utterance] = []
    with mp.Pool(processes=NUM_WORKERS) as pool, tqdm(desc='Processing utterances') as pbar:
        for u, w in pool.imap_unordered(process_utterance, load_from_prepared()):
            uttrs.extend([u] * len(w))
            words.extend(w)
            pbar.update(1)

    utt2words = defaultdict(list)
    with tqdm(desc='Compiling quantized features', total=len(words)) as pbar:
        for u, w in zip(uttrs, quantize_features(words)):
            utt2words[u].append(w)
            pbar.update(1)

    model, tokenizer = bert.load()
    os.makedirs(FEAT_DIR, exist_ok=True)
    for u, ws in tqdm(utt2words.items(), total=len(utt2words), desc='Writing files to disk'):
        ws.sort(key=lambda word: word.index)
        tokembs, tokspan = bert.embed(u.text, model, tokenizer)
        q = QSample(
            symbols=np.asarray(encode_text(u.text), dtype=np.uint8),
            tokembs=tokembs,
            tokspan=tokspan,
            wrdspan=compute_word_spans(u.text, ws),
            qfeatures=np.asarray([w.feats for w in ws]).astype(np.uint8),
        )
        q.save(os.path.join(FEAT_DIR, f'{u.filename}.h5'))


class QSample(NamedTuple):
    symbols: NDArray
    tokembs: NDArray
    tokspan: NDArray
    wrdspan: NDArray
    qfeatures: NDArray

    @classmethod
    def load(cls, path: str | os.PathLike) -> QSample:
        d = {}
        with h5py.File(path, mode='r') as h5:
            for k in cls._fields:
                d[k] = np.asarray(h5[k])
        return QSample(**d)

    def save(self, path: str | os.PathLike) -> None:
        with h5py.File(path, mode='w') as h5:
            for k, v in self._asdict().items():
                h5.create_dataset(k, data=v)


if __name__ == '__main__':
    tyro.cli(Union[download, prepare, process], description=__doc__)
