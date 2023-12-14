"""Download and preprocess LJSpeech dataset for q-VITS2 training."""

import os
import sys

from typing import Iterable, NamedTuple, Union

sys.path.append(os.path.dirname(__file__))

import requests
import tyro

from tqdm import tqdm

from utils.text import clean
from . import consts


URL = 'https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2'
DIR = 'LJSpeech-1.1'


class Sample(NamedTuple):
    filename: str
    text: str


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


def load(filename: str = 'metadata.csv') -> Iterable[Sample]:
    """Loads dataset samples lazily."""

    filepath = os.path.join(consts.DATA_DIR, DIR, filename)
    with open(filepath, mode='r', encoding='utf-8') as file:
        for line in file:
            name, _, text = line.split('|')
            yield Sample(name, clean(text.strip()))


def download() -> None:
    """Downloads the LJSpeech dataset to data directory."""

    os.makedirs(consts.DATA_DIR, exist_ok=True)

    # download the dataset unless it already exists
    filename = os.path.join(consts.DATA_DIR, os.path.basename(URL))
    if not os.path.exists(filename):
        print(f'Downloading {URL} to {filename}...')
        download_file(URL, filename)
    else:
        print(f'{filename} already exists, skipping download...')

    # unpack the tar.bz2 file into ljspeech directory
    data_dir = os.path.join(consts.DATA_DIR, DIR)
    if not os.path.exists(data_dir):
        print(f'Unpacking {filename}...')
        if os.system(f'tar -xjf {filename} -C {consts.DATA_DIR}'):
            raise IOError(f'Error while extracting data!')
    else:
        print(f'{data_dir} already exists, skipping unpacking...')

    print('Download complete.')


def preprocess() -> None:
    """Prepares all the necessary files for training."""

    wavs_dir = os.path.join(consts.DATA_DIR, DIR, 'wavs')
    for sample in load():
        path = os.path.join(wavs_dir, f'{sample.filename}.txt')
        with open(path, mode='w', encoding='utf-8') as file:
            file.write(sample.text + '\n')


if __name__ == '__main__':
    tyro.cli(Union[download, preprocess], description=__doc__)
