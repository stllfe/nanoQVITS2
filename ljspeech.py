"""Download and preprocess LJSpeech dataset for qVITS2 training."""

from __future__ import annotations

import multiprocessing as mp
import os
import sys

from collections import defaultdict
from pathlib import Path
from typing import Iterable


sys.path.append(os.path.dirname(__file__))

import numpy as np
import requests
import tyro

from textgrid import TextGrid
from tqdm import tqdm

from utils import bert
from utils.audio import normalize
from utils.audio import readwav
from utils.audio import writewav
from utils.data import Sample
from utils.data import Utterance
from utils.qfeatures import Word
from utils.qfeatures import compute_word_features
from utils.qfeatures import compute_word_spans
from utils.qfeatures import quantize_features
from utils.text import clean
from utils.text import encode_text


TEXT_MIN_LENGTH = 1
TEXT_MAX_LENGTH = 250

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
    with (
        open(fname, mode='wb') as file,
        tqdm(
            desc=fname,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
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
            raise IOError('Error while extracting data!')
    else:
        print(f'{data_dir} already exists, skipping unpacking...')

    print('Download complete.')


def prepare(dst: str | None = None) -> None:
    """Prepares all the necessary files for alignment."""

    wavs_dir = Path(DATA_DIR, DIR, 'wavs')

    for ut in tqdm(load_from_metadata(), desc='Preparing samples'):
        wavpath = Path(wavs_dir, ut.filename).with_suffix('.wav')
        try:
            wav, sr = readwav(wavpath)
        except FileNotFoundError:
            tqdm.write(f'Not found: {wavpath}', sys.stderr)
            continue
        writewav(normalize(wav), sr, wavpath)

        labpath = Path(wavs_dir, ut.filename).with_suffix('.lab')
        with open(labpath, mode='w', encoding='utf-8') as file:
            file.write(clean(ut.text) + '\n')


def process() -> None:
    """Extracts all features for training and stores them on disk."""

    words: list[Word] = []
    uttrs: list[Utterance] = []
    with mp.Pool(processes=NUM_WORKERS) as pool, tqdm(desc='Processing utterances') as pbar:
        for ut, w in pool.imap_unordered(process_utterance, load_from_prepared()):
            uttrs.extend([ut] * len(w))
            words.extend(w)
            pbar.update(1)

    ut2ws: dict[Utterance, list[Word]] = defaultdict(list)
    with tqdm(desc='Compiling quantized features', total=len(words)) as pbar:
        for ut, w in zip(uttrs, quantize_features(words)):
            ut2ws[ut].append(w)
            pbar.update(1)

    model, tokenizer = bert.load()
    os.makedirs(FEAT_DIR, exist_ok=True)
    for ut, ws in tqdm(ut2ws.items(), total=len(ut2ws), desc='Writing files to disk'):
        ws.sort(key=lambda word: word.index)
        tokembs, tokspan = bert.embed(ut.text, model, tokenizer)
        symbols = encode_text(ut.text)
        if TEXT_MIN_LENGTH > len(symbols) > TEXT_MAX_LENGTH:
            tqdm.write(f'Skip due to text length: {ut.filename}', sys.stderr)
            continue
        q = Sample(
            symbols=np.asarray(symbols, dtype=np.uint8),
            tokembs=tokembs,
            tokspan=tokspan,
            wrdspan=compute_word_spans(ut.text, ws),
            qfeatures=np.asarray([w.feats for w in ws]).astype(np.uint8),
        )
        q.save(Path(FEAT_DIR, ut.filename).with_suffix('.h5'))


def process_utterance(uttr: Utterance) -> tuple[Utterance, list[Word]]:
    wavpath = Path(WAVS_DIR, uttr.filename).with_suffix('.wav')
    mfapath = Path(MFAS_DIR, uttr.filename).with_suffix('.TextGrid')

    audio, rate = readwav(wavpath.as_posix())
    alignment = TextGrid.fromFile(mfapath.as_posix())
    words = compute_word_features(audio, alignment, rate=rate, min_duration=0.01)

    return uttr, list(words)


if __name__ == '__main__':
    tyro.extras.subcommand_cli_from_dict(
        {
            'download': download,
            'prepare': prepare,
            'process': process,
        },
        description=__doc__,
    )
