from __future__ import annotations

import operator as op
import os
import random
import sys

from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, NamedTuple, TypeVar

import h5py
import numpy as np
import torch
import torch.utils.data

from numpy.typing import NDArray
from torch import FloatTensor
from torch import LongTensor
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from config import AudioConfig
from config import TextConfig
from utils import helpers
from utils.audio import MAX_WAV_VALUE
from utils.audio import readwav
from utils.helpers import debug
from vits2 import commons
from vits2.mel_processing import mel_spectrogram_torch
from vits2.mel_processing import spectrogram_torch


WaveForm = FloatTensor
SpecForm = FloatTensor


class Utterance(NamedTuple):
    """A single speech data entity."""

    filename: str
    text: str


T = TypeVar('T', Tensor, NDArray)


@dataclass(slots=True)
class Features(Generic[T]):
    """The features extracted from an :class:`Utterance`."""

    symbols: T
    qlabels: T
    bert_embeds: T
    bert_spans: T
    word_spans: T

    @classmethod
    def load(cls, path: str | os.PathLike) -> Features:
        d = {}
        with h5py.File(path, mode='r') as h5:
            for k in cls.__slots__:
                d[k] = np.asarray(h5[k])
        return Features(**d)

    def save(self, path: str | os.PathLike) -> None:
        x = self.numpy()
        with h5py.File(path, mode='w') as h5:
            for k, v in asdict(x).items():
                h5.create_dataset(k, data=v)

    def numpy(self) -> Features[NDArray]:
        d = asdict(self)
        for k, v in d.items():
            d[k] = v if isinstance(v, np.ndarray) else v.detach().cpu().numpy()
            assert isinstance(d[k], np.ndarray)
        return Features(**d)

    def torch(self, device: str | torch.device | None = None, pin=False) -> Features[Tensor]:
        d = asdict(self)
        should_pin = torch.cuda.is_available() and str(device).startswith('cuda') and pin
        for k, v in d.items():
            # torch doesn't support unsigned tensors unfortunately
            v = v.astype(np.int32) if v.dtype in (np.uint32, np.uint8) else v
            v = torch.from_numpy(v) if isinstance(v, np.ndarray) else v
            assert isinstance(v, torch.Tensor)
            if device is not None and should_pin:
                v = v.pin_memory(device)
            d[k] = v.to(device, non_blocking=True)
        return Features(**d)


# TODO return to jaxtyping or something similar
# variable axes:
# num_words
#   word_spans
#   gt_labels
# num_tokens
#   token_spans
#   bert_embeds
# num_chars
#   text
# wave_length
#   wave
# spec_length
#   spec


# @dataclass(slots=True, eq=False)  # TODO: investigate why using dataclasses messes up tensor devices
class BatchPadded(NamedTuple):
    """A batch of pad-collated samples."""

    text: LongTensor
    text_lengths: LongTensor

    spec: SpecForm
    spec_lengths: LongTensor

    wave: WaveForm
    wave_lengths: LongTensor

    qlabels: LongTensor
    word_lengths: LongTensor

    bert_embeds: FloatTensor
    bert_lengths: LongTensor

    bert_spans: LongTensor
    word_spans: LongTensor

    def items(self) -> Iterable[tuple[str, Tensor]]:
        return self._asdict().items()


def load_filename_list(list_path: str | os.PathLike) -> list[str]:
    """Loads a list of filenames to select a subset of data."""

    filenames: list[str] = []
    with open(list_path, encoding='utf-8', mode='r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            filenames.append(line)
    return filenames


class TTSDataset(torch.utils.data.Dataset):
    """Dataset that samples (mel) spectrograms, waveforms and qfeatures."""

    def __init__(
        self,
        audio_config: AudioConfig,
        text_config: TextConfig | None,
        filenames: Sequence[str],
        feat_dir: str | os.PathLike,
        wavs_dir: str | os.PathLike,
        cache: bool = True,
        seed: int | None = 1234,
    ) -> None:
        self._audio_config = audio_config
        self._text_config = text_config
        self._cache = cache
        self._feat_dir = Path(feat_dir)
        self._wavs_dir = Path(wavs_dir)
        self._filenames = filenames
        self._samples: list[tuple[Path, Path]] = []
        self._lengths: list[int] = []

        self._find_samples()
        random.seed(seed)
        random.shuffle(self._samples)
        self._compute_lengths()

    def _find_samples(self) -> None:
        """Finds wavs and features paths with matching filennames."""

        self._samples.clear()
        for filename in tqdm(self._filenames, desc='Looking for files', disable=helpers.DEBUG < 1):
            wp = Path(self._wavs_dir, filename).with_suffix('.wav')
            fp = Path(self._feat_dir, filename).with_suffix('.h5')
            if not wp.exists():
                tqdm.write(f'Skip, no audio: {filename}', sys.stderr)
                continue
            if not fp.exists():
                tqdm.write(f'Skip, no features: {filename}', sys.stderr)
                continue
            self._samples.append((wp, fp))
        assert self._samples, 'No samples found!'

    def _compute_lengths(self) -> None:
        """Computes spectrogram lengths for bucketing."""

        # store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        self._lengths.clear()
        for wp, _ in tqdm(self._samples, desc='Calculating spec lengths'):
            length = os.path.getsize(wp) // (2 * self._audio_config.hop_length)
            self._lengths.append(length)

    @property
    def lengths(self) -> list[int]:
        return self._lengths

    def get_feats(self, featpath: str | os.PathLike) -> Features:
        feat = Features.load(featpath)
        if self._text_config and self._text_config.add_blank:
            feat.symbols = commons.intersperse(feat.symbols, 0)
        return feat

    def get_audio(self, wavpath: str | os.PathLike) -> tuple[SpecForm, WaveForm]:
        wavpath = Path(wavpath)

        wave, sampling_rate = readwav(wavpath)
        if sampling_rate != self._audio_config.sampling_rate:
            # TODO: maybe resample then?
            raise ValueError(f"{sampling_rate} SR doesn't match target {self.sampling_rate} SR")

        wave = torch.from_numpy(wave)
        wave = wave / MAX_WAV_VALUE
        wave = wave.unsqueeze(0)

        # TODO: clip values here or normalize straight to [-1, 1] range?
        debug(f'{wavpath}: min={wave.min():.4f} max={wave.max():.4f} dtype={wave.dtype}', level=3)

        specpath = wavpath.with_suffix('.npy')
        if specpath.exists():
            debug(f'Cache hit: {specpath}', level=2)
            spec = np.load(specpath)
            return torch.from_numpy(spec), torch.squeeze(wave, 0)
        if self._audio_config.mel:
            # TODO: if linear spec exists convert to mel from existing linear spec
            spec = mel_spectrogram_torch(
                wave,
                n_fft=self._audio_config.filter_length,
                num_mels=self._audio_config.mel.num_channels,
                sampling_rate=self._audio_config.sampling_rate,
                hop_size=self._audio_config.hop_length,
                win_size=self._audio_config.win_length,
                fmin=self._audio_config.mel.fmin,
                fmax=self._audio_config.mel.fmax,
                center=False,
            )
        else:
            spec = spectrogram_torch(
                wave,
                n_fft=self._audio_config.filter_length,
                sampling_rate=self._audio_config.sampling_rate,
                hop_size=self._audio_config.hop_length,
                win_size=self._audio_config.win_length,
                center=False,
            )

        spec = torch.squeeze(spec, 0)
        wave = torch.squeeze(wave, 0)
        if self._cache:
            debug(f'Cached: {specpath}', level=3)
            np.save(specpath, spec.numpy())
        return spec, wave

    def __getitem__(self, index: int) -> tuple[Features, SpecForm, WaveForm]:
        wp, fp = self._samples[index]
        spec, wave = self.get_audio(wp)
        feat = self.get_feats(fp)
        feat = feat.torch()
        debug(f'{wp.stem} loaded', level=2, rank=0)
        return feat, spec, wave

    def __len__(self) -> int:
        return len(self._samples)


def collate_fn(batch: list[tuple[Features, SpecForm, WaveForm]]) -> BatchPadded:
    """Collates dataset features in batched tensors of equal length."""

    feats, specs, waves = zip(*batch)

    texts = tuple(f.symbols for f in feats)
    text_padded = pad_sequence(texts, batch_first=True).long()
    text_lengths = torch.as_tensor([t.size(0) for t in texts], dtype=torch.long)

    spec_padded = pad_sequence([s.T for s in specs], batch_first=True).transpose_(2, 1)
    spec_lengths = torch.as_tensor([s.size(1) for s in specs], dtype=torch.long)

    # make waves at least 3D like mels
    wave_padded = pad_sequence(waves, batch_first=True).unsqueeze_(1)
    wave_lengths = torch.as_tensor([w.size(0) for w in waves], dtype=torch.long)

    # pad bert embeddings
    # TODO: maybe save as flat already?
    bert_embeds = tuple(f.bert_embeds.squeeze(0) for f in feats)
    bert_embeds_padded = pad_sequence(bert_embeds, batch_first=True)
    bert_lengths = torch.as_tensor([e.size(0) for e in bert_embeds], dtype=torch.long)

    # pad qlabels and extract word lengths
    qlabels = tuple(f.qlabels for f in feats)
    qlabels_padded = pad_sequence(qlabels, batch_first=True)
    word_lengths = torch.as_tensor([q.size(0) for q in qlabels], dtype=torch.long)

    # these lengths we already know
    bert_spans = pad_sequence(tuple(f.bert_spans for f in feats), batch_first=True)
    word_spans = pad_sequence(tuple(f.word_spans for f in feats), batch_first=True)

    # order items by spec lengths in decreasing order
    indices = torch.argsort(spec_lengths, descending=True)
    reorder = op.itemgetter(indices)

    return BatchPadded(
        text=reorder(text_padded),
        text_lengths=reorder(text_lengths),
        spec=reorder(spec_padded),
        spec_lengths=reorder(spec_lengths),
        wave=reorder(wave_padded),
        wave_lengths=reorder(wave_lengths),
        qlabels=reorder(qlabels_padded),
        word_lengths=reorder(word_lengths),
        bert_embeds=reorder(bert_embeds_padded),
        bert_lengths=reorder(bert_lengths),
        bert_spans=reorder(bert_spans),
        word_spans=reorder(word_spans),
    )


class DistributedBucketSampler(torch.utils.data.DistributedSampler):
    """Maintains similar input lengths in a batch.

    Length groups are specified by boundaries.

    For example:
        boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    I.e. boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        lengths: Sequence[int],
        boundaries: Sequence[int],
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
    ) -> None:
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = lengths
        self.batch_size = batch_size
        self.boundaries = boundaries  # [32, 300, 400, 500, 600, 700, 800, 900, 1000]

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self) -> tuple[list[int], list[int]]:
        buckets: list[int] = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket: list[int] = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[: (rem % len_bucket)]
            )
            """
            if len_bucket > 0: #- fix for  https://github.com/FENRlR/MB-iSTFT-VITS2/issues/9
                rem = num_samples_bucket - len_bucket
                ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
            """

            # subsample
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[j * self.batch_size : (j + 1) * self.batch_size]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x: int, lo=0, hi: int | None = None) -> int:
        if hi is None:
            hi = len(self.boundaries) - 1
        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        return -1

    def __len__(self) -> int:
        return self.num_samples // self.batch_size
