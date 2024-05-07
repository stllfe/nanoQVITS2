import os

from dataclasses import dataclass
from os import PathLike
from typing import Literal

from ljspeech import FEAT_DIR
from ljspeech import ROOT_DIR
from ljspeech import WAVS_DIR


PROJ_DIR = os.path.abspath(os.path.dirname(__file__))
LOGS_DIR = os.path.join(PROJ_DIR, 'logs')


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
    resblock: Literal['1', '2']
    resblock_kernel_sizes: tuple[int, int, int]
    resblock_dilation_sizes: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]
    upsample_rates: tuple[int, int]
    upsample_initial_channel: 256
    upsample_kernel_sizes: tuple[int, int]
    n_layers_q: int = 3
    use_spectral_norm: bool = False
    use_sdp: bool = False


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
        # a template, or get from somewhere?
        model_name = 'mini-mb-istft-vits2_LJSpeech'
        self.model_dir = os.path.join(LOGS_DIR, model_name)


config = ExperimentConfig(
    data=DataConfig(
        wavs_dir=WAVS_DIR,
        feat_dir=FEAT_DIR,
        train_list_path=os.path.join(ROOT_DIR, 'train.list'),
        valid_list_path=os.path.join(ROOT_DIR, 'valid.list'),
        text=TextConfig(add_blank=True),
        audio=AudioConfig(
            sampling_rate=22050,
            filter_length=1024,
            hop_length=256,
            win_length=1024,
            mel=MelSpecConfig(
                num_channels=80,
            ),
        ),
    ),
    model=ModelConfig(
        use_mel_posterior_encoder=True,
        use_transformer_flows=True,
        transformer_flow_type='pre_conv2',
        use_spk_conditioned_encoder=False,
        use_noise_scaled_mas=True,
        use_duration_discriminator=True,
        duration_discriminator_type='dur_disc_2',
        ms_istft_vits=False,
        mb_istft_vits=True,
        istft_vits=False,
        subbands=4,
        gen_istft_n_fft=16,
        gen_istft_hop_size=4,
        inter_channels=192,
        hidden_channels=96,
        filter_channels=768,
        n_heads=2,
        n_layers=3,
        kernel_size=3,
        p_dropout=0.1,
        resblock='1',
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        upsample_rates=(4, 4),
        upsample_initial_channel=256,
        upsample_kernel_sizes=(16, 16),
        n_layers_q=3,
        use_spectral_norm=False,
        use_sdp=False,
    ),
    train=TrainConfig(
        log_interval=200,
        eval_interval=1000,
        seed=1234,
        epochs=20000,
        learning_rate=2e-4,
        betas=(0.8, 0.99),
        eps=1e-9,
        batch_size=32,
        fp16_run=False,
        lr_decay=0.999875,
        segment_size=8192,
        init_lr_ratio=1,
        warmup_epochs=0,
        c_mel=45,
        c_kl=1.0,
        fft_sizes=(384, 683, 171),
        hop_sizes=(30, 60, 10),
        win_lengths=(150, 300, 60),
        window='hann_window',
    ),
)
