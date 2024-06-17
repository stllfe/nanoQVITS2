"""Download and preprocess LJSpeech dataset for qVITS2 training."""

from __future__ import annotations

import multiprocessing as mp
import os
import random
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
from utils.audio import readwav
from utils.data import Features
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
DIRNAME = 'LJSpeech-1.1'

ROOT_DIR = os.path.join(DATA_DIR, DIRNAME)
WAVS_DIR = os.path.join(ROOT_DIR, 'wavs')
MFAS_DIR = os.path.join(ROOT_DIR, 'alignment')
FEAT_DIR = os.path.join(ROOT_DIR, 'features')


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
        ) as pbar,
    ):
        for data in response.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            pbar.update(size)


def load_from_metadata(filename: str = 'metadata.csv') -> Iterable[Utterance]:
    """Loads dataset samples lazily."""

    filepath = os.path.join(ROOT_DIR, filename)
    with open(filepath, mode='r', encoding='utf-8') as file:
        for line in file:
            name, _, text = line.split('|')
            yield Utterance(name, text.strip())


def load_from_prepared() -> Iterable[Utterance]:
    """Loads dataset samples from disk."""

    wavs_dir = Path(ROOT_DIR, 'wavs')
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
    data_dir = os.path.join(DATA_DIR, DIRNAME)

    # TODO: can we skip it and do the prepare step from the compressed tar,
    # just by sequentially reading it?
    if not os.path.exists(data_dir):
        print(f'Unpacking {filename}...')
        if os.system(f'tar -xjf {filename} -C {DATA_DIR}'):
            raise IOError('Error while extracting data!')
    else:
        print(f'{data_dir} already exists, skipping unpacking...')

    print('Download complete.')


def prepare() -> None:
    """Prepares all the necessary files for alignment."""

    for ut in tqdm(load_from_metadata(), desc='Preparing samples'):
        wavpath = Path(WAVS_DIR, ut.filename).with_suffix('.wav')
        try:
            readwav(wavpath)
        except FileNotFoundError:
            tqdm.write(f'Not found: {wavpath}', sys.stderr)
            continue
        labpath = Path(WAVS_DIR, ut.filename).with_suffix('.lab')
        with open(labpath, mode='w', encoding='utf-8') as file:
            file.write(clean(ut.text) + '\n')


def process() -> None:
    """Extracts all features for training and stores them on disk."""

    words: list[Word] = []
    uttrs: list[Utterance] = []

    buff: list[Utterance] = list(load_from_prepared())
    with (
        mp.Pool(processes=NUM_WORKERS) as pool,
        tqdm(desc='Processing utterances', total=len(buff)) as pbar,
    ):
        for ut, w in pool.imap_unordered(process_utterance, buff):
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
        symbols = encode_text(ut.text)
        if TEXT_MIN_LENGTH > len(symbols) > TEXT_MAX_LENGTH:
            tqdm.write(f'Skip due to text length: {ut.filename}', sys.stderr)
            continue
        bert_embeds, bert_spans = bert.embed(ut.text, model, tokenizer)
        feat = Features(
            symbols=np.asarray(symbols, dtype=np.uint8),
            qlabels=np.asarray([w.feats for w in ws]).astype(np.uint8),
            bert_embeds=bert_embeds,
            bert_spans=bert_spans,
            word_spans=compute_word_spans(ut.text, ws),
        )
        feat.save(Path(FEAT_DIR, ut.filename).with_suffix('.h5'))


def process_utterance(ut: Utterance) -> tuple[Utterance, list[Word]]:
    wavpath = Path(WAVS_DIR, ut.filename).with_suffix('.wav')
    mfapath = Path(MFAS_DIR, ut.filename).with_suffix('.TextGrid')

    audio, rate = readwav(wavpath)
    alignment = TextGrid.fromFile(mfapath)
    words = compute_word_features(audio, alignment, rate=rate, min_duration=0.01)

    return ut, list(words)


def split(num_test: int = 500, num_valid: int = 100, seed: int = 25512) -> None:
    """Compiles train, valid, test file lists for the processed dataset."""

    def exists(ut: Utterance) -> bool:
        wp = Path(WAVS_DIR, ut.filename).with_suffix('.wav')
        fp = Path(FEAT_DIR, ut.filename).with_suffix('.h5')
        return wp.exists() and fp.exists()

    uttrs = sorted(filter(exists, load_from_prepared()))
    index = range(len(uttrs))

    random.seed(seed)
    test = random.sample(index, k=num_test)
    left = set(index) - set(test)

    valid = random.sample(sorted(left), k=num_valid)
    train = sorted(left - set(valid))

    for subset, indices in (('test', test), ('valid', valid), ('train', train)):
        filepath = Path(ROOT_DIR, subset).with_suffix('.list')
        with open(filepath, mode='w', encoding='utf-8') as file:
            for i in indices:
                file.write(uttrs[i].filename + '\n')


if __name__ == '__main__':
    tyro.extras.subcommand_cli_from_dict(
        description=__doc__,
        subcommands={
            'download': download,
            'prepare': prepare,
            'process': process,
            'split': split,
        },
    )
