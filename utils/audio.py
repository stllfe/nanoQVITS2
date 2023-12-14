import numpy as np
import pyreaper

from numpy.typing import NDArray
from scipy.io import wavfile
from scipy import stats


MIN_F0 = 40
MAX_F0 = 1100


def read(wavpath: str, mmap: bool = False) -> tuple[NDArray, int]:
    return wavfile.read(wavpath, mmap=mmap)[::-1]


def compute_pitch(audio: NDArray, rate: int, freq: float = 0.01) -> tuple[NDArray, NDArray]:
    """Computes pitch (F0) values for the given audio.

    Returns:
        A tuple of time frames in seconds and corresponding f0 values.
    """

    outs = pyreaper.reaper(
        audio,
        rate,
        minf0=MIN_F0,
        maxf0=MAX_F0,
        frame_period=freq,
        unvoiced_cost=0.9
    )
    return outs[2:4]


def compute_pitch_slope(f0: NDArray) -> float:
    x = np.arange(len(f0))
    mask = f0 > 0
    if len(f0[mask]):
        return stats.linregress(x[mask], f0[mask])[0]
    return 0
