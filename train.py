import os
import sys

from dataclasses import asdict
from logging import Logger
from pprint import pformat
from typing import Iterable

import psutil
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm

from torch import nn
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import vits2.commons as commons
from vits2.qwfp import compute_q_loss
import vits2.utils as utils

from config import AudioConfig
from config import DataConfig
from config import ExperimentConfig
from config import MelSpecConfig
from config import ModelConfig
from config import TextConfig
from config import TrainConfig
from ljspeech import FEAT_DIR
from ljspeech import ROOT_DIR
from ljspeech import WAVS_DIR
from utils.data import BatchPadded
from utils.data import DistributedBucketSampler
from utils.data import TTSDataset
from utils.data import collate_fn
from utils.data import load_filename_list
from utils.text import SYMBOLS
from vits2.losses import discriminator_loss
from vits2.losses import feature_loss
from vits2.losses import generator_loss
from vits2.losses import kl_loss
from vits2.losses import subband_stft_loss
from vits2.mel_processing import mel_spectrogram_torch
from vits2.mel_processing import spec_to_mel_torch
from vits2.models import AVAILABLE_DURATION_DISCRIMINATOR_TYPES
from vits2.models import AVAILABLE_FLOW_TYPES
from vits2.models import DurationDiscriminator
from vits2.models import DurationDiscriminator2
from vits2.models import MultiPeriodDiscriminator
from vits2.models import SynthesizerTrn
from vits2.pqmf import PQMF


ModuleOrDDP = nn.Module | DDP

torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
global_step = 0

CONFIG = ExperimentConfig(
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
        n_speakers=0,
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
        q_condition_layer=2,
    ),
    train=TrainConfig(
        log_interval=200,
        eval_interval=1000,
        seed=1234,
        epochs=20000,
        learning_rate=2e-4,
        betas=(0.8, 0.99),
        eps=1e-9,
        batch_size=16,
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

# FIXME: training raises  warning:
# Grad strides do not match bucket view strides ...
# Check this: https://github.com/pytorch/pytorch/issues/47163
# Seems like transpose operations might cause some issues, need to investigate


def ismaster(rank: int) -> bool:
    return rank == 0


# - base vits2 : Aug 29, 2023
def main() -> None:
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), 'CPU training is not allowed.'

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '6060'

    cfg = CONFIG
    print(f'Detected GPUs: {n_gpus}')
    if n_gpus > 1:
        print('Multi GPU training starting...')
        mp.spawn(run, nprocs=n_gpus, args=(n_gpus, cfg))
    else:
        print('Single GPU training starting...')
    run(0, n_gpus=1, cfg=cfg)


def run(rank: int, n_gpus: int, cfg: ExperimentConfig) -> None:
    global global_step

    if ismaster(rank):
        logger = utils.get_logger(cfg.model_dir)
        logger.info(pformat(cfg))
        utils.check_git_hash(cfg.model_dir)
        # TODO: add wandb option
        writer = SummaryWriter(log_dir=cfg.model_dir)
    else:
        # FIXME: are there any better solutions?
        logger = None
        writer = None

    if os.name == 'nt':
        dist.init_process_group(backend='gloo', init_method='env://', world_size=n_gpus, rank=rank)
    else:
        dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)

    torch.manual_seed(cfg.train.seed)
    torch.cuda.set_device(rank)

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

    train_dataset = TTSDataset(
        audio_config=cfg.data.audio,
        text_config=cfg.data.text,
        filenames=load_filename_list(cfg.data.train_list_path),
        feat_dir=cfg.data.feat_dir,
        wavs_dir=cfg.data.wavs_dir,
        cache=True,
        seed=cfg.train.seed,
    )
    train_sampler = DistributedBucketSampler(
        train_dataset,
        batch_size=cfg.train.batch_size,
        lengths=train_dataset.lengths,
        boundaries=[32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    train_loader = DataLoader(
        train_dataset,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        # TODO: looks like this produces a memory leak, investigate:
        # find a way to track RAM in tensorboard/wandb
        # persistent_workers=True,
        prefetch_factor=8,
    )
    if ismaster(rank):
        eval_dataset = TTSDataset(
            audio_config=cfg.data.audio,
            text_config=cfg.data.text,
            filenames=load_filename_list(cfg.data.valid_list_path),
            feat_dir=cfg.data.feat_dir,
            wavs_dir=cfg.data.wavs_dir,
            cache=True,
            seed=cfg.train.seed,
        )
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=1,
            shuffle=False,
            batch_size=cfg.train.batch_size,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
    else:
        eval_loader = None

    # some of these flags are not being used in the code and directly set in hps json file.
    # they are kept here for reference and prototyping.

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
            print(
                'Warning: `use_spk_conditioned_encoder` is True but `n_speakers` is 0', sys.stderr
            )
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
            duration_discriminator_type in AVAILABLE_DURATION_DISCRIMINATOR_TYPES.keys()
        ), f"Unknown duration discriminator type '{duration_discriminator_type}'! Should be one of {AVAILABLE_DURATION_DISCRIMINATOR_TYPES}"
        # DurationDiscriminator = AVAILABLE_DURATION_DISCRIMINATOR_TYPES[duration_discriminator_type]

        if duration_discriminator_type == 'dur_disc_1':
            net_dur_disc = DurationDiscriminator(
                in_channels=cfg.model.hidden_channels,
                filter_channels=cfg.model.hidden_channels,
                kernel_size=3,
                p_dropout=0.1,
                gin_channels=cfg.model.gin_channels if cfg.data.n_speakers != 0 else 0,
            ).cuda(rank)
        elif duration_discriminator_type == 'dur_disc_2':
            net_dur_disc = DurationDiscriminator2(
                in_channels=cfg.model.hidden_channels,
                filter_channels=cfg.model.hidden_channels,
                kernel_size=3,
                p_dropout=0.1,
                gin_channels=cfg.model.gin_channels if cfg.data.n_speakers != 0 else 0,
            ).cuda(rank)
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
    ).cuda(rank)
    # net_g = torch.compile(net_g)
    # logger.info('Generator model compiled!')
    net_d = MultiPeriodDiscriminator(cfg.model.use_spectral_norm).cuda(rank)
    # net_d = torch.compile(net_d)
    # logger.info('Discriminator model compiled!')

    optim_g = torch.optim.AdamW(
        net_g.parameters(), cfg.train.learning_rate, betas=cfg.train.betas, eps=cfg.train.eps
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(), cfg.train.learning_rate, betas=cfg.train.betas, eps=cfg.train.eps
    )

    if net_dur_disc is not None:
        optim_dur_disc = torch.optim.AdamW(
            net_dur_disc.parameters(),
            cfg.train.learning_rate,
            betas=cfg.train.betas,
            eps=cfg.train.eps,
        )
    else:
        optim_dur_disc = None

    net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)

    if net_dur_disc is not None:  # 2의 경우
        net_dur_disc = DDP(net_dur_disc, device_ids=[rank], find_unused_parameters=True)

    try:
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(cfg.model_dir, 'G_*.pth'), net_g, optim_g
        )
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(cfg.model_dir, 'D_*.pth'), net_d, optim_d
        )
        if net_dur_disc is not None:  # 2의 경우
            _, _, _, epoch_str = utils.load_checkpoint(
                utils.latest_checkpoint_path(cfg.model_dir, 'DUR_*.pth'),
                net_dur_disc,
                optim_dur_disc,
            )
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        # FIXME: need to catch a concrete exception here
        epoch_str = 1
        global_step = 0

    scheduler_g = ExponentialLR(optim_g, gamma=cfg.train.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d = ExponentialLR(optim_d, gamma=cfg.train.lr_decay, last_epoch=epoch_str - 2)
    if net_dur_disc is not None:  # 2의 경우
        scheduler_dur_disc = ExponentialLR(
            optim_dur_disc, gamma=cfg.train.lr_decay, last_epoch=epoch_str - 2
        )
    else:
        scheduler_dur_disc = None

    scaler = GradScaler(enabled=cfg.train.fp16_run)

    for epoch in range(epoch_str, cfg.train.epochs + 1):
        train_and_evaluate(
            rank,
            epoch,
            cfg,
            [net_g, net_d, net_dur_disc],
            [optim_g, optim_d, optim_dur_disc],
            scaler,
            [train_loader, eval_loader],
            logger,
            writer,
        )
        scheduler_g.step()
        scheduler_d.step()
        if net_dur_disc is not None:
            scheduler_dur_disc.step()


def train_and_evaluate(
    rank: int,
    epoch: int,
    cfg: ExperimentConfig,
    nets: tuple[ModuleOrDDP, ModuleOrDDP, ModuleOrDDP],
    optims: tuple[Optimizer, Optimizer, Optimizer],
    scaler: GradScaler,
    loaders: tuple[DataLoader, DataLoader | None],
    logger: Logger | None,
    writer: SummaryWriter | None,
) -> None:
    net_g, net_d, net_dur_disc = nets
    optim_g, optim_d, optim_dur_disc = optims
    train_loader, eval_loader = loaders

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()

    if net_dur_disc is not None:  # vits2
        net_dur_disc.train()

    process = psutil.Process()
    loader: Iterable[BatchPadded] = tqdm.tqdm(
        train_loader, desc=f'[EPOCH {epoch:03}] Loading training data', disable=not ismaster(rank)
    )

    for batch_idx, batch in enumerate(loader):
        if net_g.module.use_noise_scaled_mas:
            current_mas_noise_scale = (
                net_g.module.mas_noise_scale_initial - net_g.module.noise_scale_delta * global_step
            )
            net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)
        # TODO: add a helper function or method (better a combination) for moving BatchPadded tensors to appropriate device
        # TODO: make model accept a single data structure without the need for passing everything as a separate argument
        x, x_lengths = (
            batch.text.cuda(rank, non_blocking=True),
            batch.text_lengths.cuda(rank, non_blocking=True),
        )
        spec, spec_lengths = (
            batch.spec.cuda(rank, non_blocking=True),
            batch.spec_lengths.cuda(rank, non_blocking=True),
        )
        y, y_lengths = (
            batch.wave.cuda(rank, non_blocking=True),
            batch.wave_lengths.cuda(rank, non_blocking=True),
        )

        with autocast(enabled=cfg.train.fp16_run):
            (
                y_hat,
                y_hat_mb,
                l_length,
                attn,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                (hidden_x, logw, logw_),
                q_logits,
            ) = net_g(x, x_lengths, spec, spec_lengths, batch)

            q_loss = compute_q_loss(q_logits, batch)

            if cfg.model.use_mel_posterior_encoder:
                mel = spec
            else:
                mel = spec_to_mel_torch(
                    # spec,
                    spec.float(),  # - for 16bit stability
                    cfg.data.audio.filter_length,
                    cfg.data.audio.mel.num_channels,
                    cfg.data.audio.sampling_rate,
                    cfg.data.audio.mel.fmin,
                    cfg.data.audio.mel.fmax,
                )
            y_mel = commons.slice_segments(
                mel, ids_slice, cfg.train.segment_size // cfg.data.audio.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                cfg.data.audio.filter_length,
                cfg.data.audio.mel.num_channels,
                cfg.data.audio.sampling_rate,
                cfg.data.audio.hop_length,
                cfg.data.audio.win_length,
                cfg.data.audio.mel.fmin,
                cfg.data.audio.mel.fmax,
            )

            y = commons.slice_segments(
                y, ids_slice * cfg.data.audio.hop_length, cfg.train.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc

            # Duration Discriminator
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                    hidden_x.detach(), x_mask.detach(), logw_.detach(), logw.detach()
                )  # logw is predicted duration, logw_ is real duration
                with autocast(enabled=False):
                    # TODO: I think need to mean using the mask, but for now, just mean all
                    loss_dur_disc, losses_dur_disc_r, losses_dur_disc_g = discriminator_loss(
                        y_dur_hat_r, y_dur_hat_g
                    )
                    loss_dur_disc_all = loss_dur_disc
                optim_dur_disc.zero_grad()
                scaler.scale(loss_dur_disc_all).backward()
                scaler.unscale_(optim_dur_disc)
                grad_norm_dur_disc = commons.clip_grad_value_(net_dur_disc.parameters(), None)
                scaler.step(optim_dur_disc)

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=cfg.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw_, logw)
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * cfg.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * cfg.train.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)

                if cfg.model.mb_istft_vits:
                    pqmf = PQMF(y.device)
                    y_mb = pqmf.analysis(y)
                    loss_subband = subband_stft_loss(cfg, y_mb, y_hat_mb)
                else:
                    loss_subband = torch.tensor(0.0)

                loss_gen_all = (
                    loss_gen + loss_fm + loss_mel + loss_dur + loss_kl + loss_subband + q_loss
                )
                if net_dur_disc is not None:
                    loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
                    loss_gen_all += loss_dur_gen

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if ismaster(rank) and global_step % cfg.train.log_interval == 0:
            lr = optim_g.param_groups[0]['lr']

            losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl, loss_subband]

            logger.info(
                'Train Epoch: {} [{:.0f}%]'.format(epoch, 100.0 * batch_idx / len(train_loader))
            )
            logger.info([x.item() for x in losses] + [global_step, lr])

            scalar_dict = {
                'mem/proc/rssMB': process.memory_info().rss / (1024**2),
                'mem/sys/availableMB': psutil.virtual_memory().available / (1024**2),
                'mem/sys/percent': psutil.virtual_memory().percent,
                'loss/g/total': loss_gen_all,
                'loss/d/total': loss_disc_all,
                'learning_rate': lr,
                'grad_norm_d': grad_norm_d,
                'grad_norm_g': grad_norm_g,
            }

            if net_dur_disc is not None:  # 2인 경우
                scalar_dict.update({
                    'loss/dur_disc/total': loss_dur_disc_all,
                    'grad_norm_dur_disc': grad_norm_dur_disc,
                })
            scalar_dict.update({
                'loss/g/fm': loss_fm,
                'loss/g/mel': loss_mel,
                'loss/g/dur': loss_dur,
                'loss/g/kl': loss_kl,
                'loss/g/subband': loss_subband,
                'loss/g/q': q_loss,
            })

            scalar_dict.update({'loss/g/{}'.format(i): v for i, v in enumerate(losses_gen)})
            scalar_dict.update({'loss/d_r/{}'.format(i): v for i, v in enumerate(losses_disc_r)})
            scalar_dict.update({'loss/d_g/{}'.format(i): v for i, v in enumerate(losses_disc_g)})

            # if net_dur_disc is not None: # - 보류?
            #   scalar_dict.update({"loss/dur_disc_r" : f"{losses_dur_disc_r}"})
            #   scalar_dict.update({"loss/dur_disc_g" : f"{losses_dur_disc_g}"})
            #   scalar_dict.update({"loss/dur_gen" : f"{loss_dur_gen}"})

            image_dict = {
                'slice/mel_org': utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                'slice/mel_gen': utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                'all/mel': utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                'all/attn': utils.plot_alignment_to_numpy(attn[0, 0].data.cpu().numpy()),
            }
            utils.summarize(
                writer=writer, global_step=global_step, images=image_dict, scalars=scalar_dict
            )

        if ismaster(rank) and global_step % cfg.train.eval_interval == 0:
            evaluate(cfg, net_g, eval_loader, writer)
            utils.save_checkpoint(
                net_g,
                optim_g,
                cfg.train.learning_rate,
                epoch,
                os.path.join(cfg.model_dir, 'G_{}.pth'.format(global_step)),
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                cfg.train.learning_rate,
                epoch,
                os.path.join(cfg.model_dir, 'D_{}.pth'.format(global_step)),
            )
            if net_dur_disc is not None:
                utils.save_checkpoint(
                    net_dur_disc,
                    optim_dur_disc,
                    cfg.train.learning_rate,
                    epoch,
                    os.path.join(cfg.model_dir, 'DUR_{}.pth'.format(global_step)),
                )

            prev_g = os.path.join(
                cfg.model_dir, 'G_{}.pth'.format(global_step - 3 * cfg.train.eval_interval)
            )
            if os.path.exists(prev_g):
                os.remove(prev_g)
                prev_d = os.path.join(
                    cfg.model_dir, 'D_{}.pth'.format(global_step - 3 * cfg.train.eval_interval)
                )
                if os.path.exists(prev_d):
                    os.remove(prev_d)
                    prev_dur = os.path.join(
                        cfg.model_dir,
                        'DUR_{}.pth'.format(global_step - 3 * cfg.train.eval_interval),
                    )
                    if os.path.exists(prev_dur):
                        os.remove(prev_dur)

        global_step += 1


def evaluate(
    cfg: ExperimentConfig,
    generator: ModuleOrDDP,
    loader: Iterable[BatchPadded],
    writer: SummaryWriter,
) -> None:
    generator.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x, x_lengths = batch.text.cuda(0), batch.text_lengths.cuda(0)
            spec, spec_lengths = batch.spec.cuda(0), batch.spec_lengths.cuda(0)
            y, y_lengths = batch.wave.cuda(0), batch.wave_lengths.cuda(0)

            # remove else
            x = x[:1]
            x_lengths = x_lengths[:1]
            spec = spec[:1]
            spec_lengths = spec_lengths[:1]
            y = y[:1]
            y_lengths = y_lengths[:1]

            # TODO: not my proudest code... add a __getitem__ method for batch?
            batch = batch._replace(**{k: v[:1].cuda(0) for k, v in batch.items()})
            break

        y_hat, y_hat_mb, attn, mask, *_ = generator.module.infer(x, x_lengths, batch, max_len=1000)
        y_hat_lengths = mask.sum([1, 2]).long() * cfg.data.audio.hop_length

        if cfg.model.use_mel_posterior_encoder:  # 2의 경우
            mel = spec
        else:
            mel = spec_to_mel_torch(
                spec=spec,
                n_fft=cfg.data.filter_length,
                num_mels=cfg.data.audio.mel.num_channels,
                sampling_rate=cfg.data.audio.sampling_rate,
                fmin=cfg.data.audio.mel.fmin,
                fmax=cfg.data.audio.mel.fmax,
            )
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1).float(),
            n_fft=cfg.data.audio.filter_length,
            num_mels=cfg.data.audio.mel.num_channels,
            sampling_rate=cfg.data.audio.sampling_rate,
            hop_size=cfg.data.audio.hop_length,
            win_size=cfg.data.audio.win_length,
            fmin=cfg.data.audio.mel.fmin,
            fmax=cfg.data.audio.mel.fmax,
        )
    image_dict = {'mel/gen': utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())}
    audio_dict = {'audio/gen': y_hat[0, :, : y_hat_lengths[0]]}
    if global_step == 0:
        image_dict.update({'mel/gt': utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
        audio_dict.update({'audio/gt': y[0, :, : y_lengths[0]]})

    utils.summarize(
        writer=writer,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=cfg.data.audio.sampling_rate,
    )
    generator.train()


if __name__ == '__main__':
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    main()
