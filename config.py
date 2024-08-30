import os

from dataclasses import dataclass
from os import PathLike
from typing import Literal


PROJ_DIR = os.path.abspath(os.path.dirname(__file__))
LOGS_DIR = os.path.join(PROJ_DIR, 'logs')


# TODO: looks like this class is a bit too much, we can fuse it with the audio config
# since we are most likely to use melspecs anyway


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
class DataConfig:
    wavs_dir: str | PathLike
    feat_dir: str | PathLike
    train_list_path: str | PathLike
    valid_list_path: str | PathLike
    text: TextConfig
    audio: AudioConfig
    n_speakers: int


# FIXME: this is heavy, need to split up and remove redundant flags
# also need to setup correct types and preconfigured model variants


@dataclass
class ModelConfig:
    use_mel_posterior_encoder: bool
    use_transformer_flows: bool
    transformer_flow_type: str
    use_spk_conditioned_encoder: bool
    use_noise_scaled_mas: bool
    use_duration_discriminator: bool
    duration_discriminator_type: str
    ms_istft_vits: bool
    mb_istft_vits: bool
    istft_vits: bool
    subbands: int
    gen_istft_n_fft: int
    gen_istft_hop_size: int
    inter_channels: int
    hidden_channels: int
    filter_channels: int
    n_heads: int
    n_layers: int
    kernel_size: int
    p_dropout: float
    resblock: Literal['1', '2']  # this is probably not enough
    resblock_kernel_sizes: tuple[int, int, int]
    resblock_dilation_sizes: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]
    upsample_rates: tuple[int, int]
    upsample_initial_channel: 256
    upsample_kernel_sizes: tuple[int, int]
    n_layers_q: int = 3
    use_spectral_norm: bool = False
    use_sdp: bool = False
    q_condition_layer: int = 2


@dataclass
class TrainConfig:
    log_interval: int
    eval_interval: int
    seed: int
    epochs: int
    learning_rate: float
    betas: tuple[float, float]
    eps: float
    batch_size: int
    fp16_run: bool
    lr_decay: float
    segment_size: int
    init_lr_ratio: float
    warmup_epochs: int
    c_mel: float
    c_kl: float
    fft_sizes: tuple[int, int, int]
    hop_sizes: tuple[int, int, int]
    win_lengths: tuple[int, int, int]
    window: str


@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig

    def __post_init__(self) -> None:
        # TODO: need a beter solution
        # a template, or get from CLI or environment?
        # default to some timecoded string?
        model_name = 'mini-mb-istft-vits2_LJSpeech'
        self.model_dir = os.path.join(LOGS_DIR, model_name)
