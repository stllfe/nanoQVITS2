from dataclasses import dataclass


@dataclass
class MelSpecConfig:
    fmin: float | None = 0
    fmax: float | None = None
    num_channels: int = 80


@dataclass
class AudioConfig:
    sampling_rate: int
    filter_length: int
    hop_length: int
    win_length: int
    mel: MelSpecConfig | None = None


@dataclass
class TextConfig:
    add_blank: bool = False


@dataclass
class FeaturesConfig:
    audio: AudioConfig
    text: TextConfig
