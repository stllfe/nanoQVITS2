import sys

from dataclasses import asdict

import torch

from torch.utils.data.dataloader import DataLoader

from train import CONFIG
from utils import helpers
from utils.data import BatchPadded
from utils.data import TTSDataset
from utils.data import collate_fn
from utils.data import load_filename_list
from utils.text import SYMBOLS
from vits2.models import AVAILABLE_DURATION_DISCRIMINATOR_TYPES
from vits2.models import AVAILABLE_FLOW_TYPES
from vits2.models import DurationDiscriminator
from vits2.models import DurationDiscriminator2
from vits2.models import SynthesizerTrn


cfg = CONFIG
torch.manual_seed(cfg.train.seed)

if cfg.model.use_mel_posterior_encoder:  # P.incoder for vits2
    print('Using mel posterior encoder for VITS2')
    assert (
        cfg.data.audio.mel
    ), "Audio doesn't use mel spectograms, but posterior encoder is enabled!"
    posterior_channels = cfg.data.audio.mel.num_channels  # vits2
else:
    print('Using lin posterior encoder for VITS1')
    posterior_channels = cfg.data.audio.filter_length // 2 + 1
    cfg.data.audio.mel = None

if cfg.model.use_transformer_flows:
    transformer_flow_type = cfg.model.transformer_flow_type
    print(f'Using transformer flows {transformer_flow_type} for VITS2')
    assert (
        transformer_flow_type in AVAILABLE_FLOW_TYPES
    ), f"Unknown flow type '{transformer_flow_type}'! Should be one of: {AVAILABLE_FLOW_TYPES}"
else:
    print('Using normal flows for VITS1')

if cfg.model.use_spk_conditioned_encoder:
    if cfg.data.n_speakers == 0:
        print('Warning: `use_spk_conditioned_encoder` is True but `n_speakers` is 0', sys.stderr)
        print('Setting `use_spk_conditioned_encoder` to False as model is single speaker...')
        cfg.model.use_spk_conditioned_encoder = False
else:
    print('Using normal encoder for VITS1 (single speaker)')

if cfg.model.use_noise_scaled_mas:
    print('Using noise scaled MAS for VITS2')
    mas_noise_scale_initial = 0.01
    noise_scale_delta = 2e-6
else:
    print('Using normal MAS for VITS1')
    mas_noise_scale_initial = 0.0
    noise_scale_delta = 0.0

if cfg.model.use_duration_discriminator:
    # print("Using duration discriminator for VITS2")

    # - for duration_discriminator2
    # duration_discriminator_type = getattr(hps.model, "duration_discriminator_type", "dur_disc_1")
    duration_discriminator_type = cfg.model.duration_discriminator_type
    print(f"Using duration discriminator '{duration_discriminator_type}' for VITS2")
    assert (
        duration_discriminator_type in AVAILABLE_DURATION_DISCRIMINATOR_TYPES
    ), f"Unknown duration discriminator type '{duration_discriminator_type}'! Should be one of {AVAILABLE_DURATION_DISCRIMINATOR_TYPES}"
    # DurationDiscriminator = AVAILABLE_DURATION_DISCRIMINATOR_TYPES[duration_discriminator_type]

    if duration_discriminator_type == 'dur_disc_1':
        net_dur_disc = DurationDiscriminator(
            in_channels=cfg.model.hidden_channels,
            filter_channels=cfg.model.hidden_channels,
            kernel_size=3,
            p_dropout=0.1,
            gin_channels=cfg.model.gin_channels if cfg.data.n_speakers != 0 else 0,
        )
    elif duration_discriminator_type == 'dur_disc_2':
        net_dur_disc = DurationDiscriminator2(
            in_channels=cfg.model.hidden_channels,
            filter_channels=cfg.model.hidden_channels,
            kernel_size=3,
            p_dropout=0.1,
            gin_channels=cfg.model.gin_channels if cfg.data.n_speakers != 0 else 0,
        )
else:
    print('NOT using any duration discriminator like VITS1')
    net_dur_disc = None

net_g = SynthesizerTrn(
    len(SYMBOLS),
    posterior_channels,
    cfg.train.segment_size // cfg.data.audio.hop_length,
    mas_noise_scale_initial=mas_noise_scale_initial,
    noise_scale_delta=noise_scale_delta,
    **asdict(cfg.model),
)

train_dataset = TTSDataset(
    audio_config=cfg.data.audio,
    text_config=cfg.data.text,
    filenames=load_filename_list(cfg.data.train_list_path),
    feat_dir=cfg.data.feat_dir,
    wavs_dir=cfg.data.wavs_dir,
    cache=True,
    seed=cfg.train.seed,
)
train_loader = DataLoader(
    train_dataset,
    batch_size=3,
    num_workers=0,
    shuffle=False,
    pin_memory=True,
    collate_fn=collate_fn,
)
with helpers.Context(DEBUG=2):
    batch: BatchPadded = next(iter(train_loader))
# print(train_dataset._samples[0])
_ = net_g(batch.text, batch.text_lengths, batch.spec, batch.spec_lengths, batch)
print(batch)
