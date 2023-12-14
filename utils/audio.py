import pyreaper

from numpy.typing import NDArray
from scipy.io import wavfile


MIN_F0 = 40
MAX_F0 = 1100


def read(wavpath: str, mmap: bool = False) -> tuple[int, NDArray]:
    return wavfile.read(wavpath, mmap=mmap)


def compute_f0(rate: int, audio: NDArray, freq: float = 0.01) -> tuple[NDArray, NDArray]:
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
