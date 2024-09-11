import re
import sys

from dataclasses import asdict

import torch

from train import CONFIG
from utils import bert
from utils.audio import writewav
from utils.data import BatchPadded
from utils.text import SYMBOLS
from utils.text import encode_text
from vits2 import commons
from vits2.models import AVAILABLE_FLOW_TYPES
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

net_g = SynthesizerTrn(
    len(SYMBOLS),
    posterior_channels,
    cfg.train.segment_size // cfg.data.audio.hop_length,
    mas_noise_scale_initial=mas_noise_scale_initial,
    noise_scale_delta=noise_scale_delta,
    **asdict(cfg.model),
)

text = 'Who do you think I am?'
text_encoded = encode_text(text)
if cfg.data.text.add_blank:
    text_encoded = commons.intersperse(text_encoded, 0)
text_encoded = torch.as_tensor(text_encoded)
text_lengths = torch.as_tensor([text_encoded.size(0)], dtype=torch.long)

bert_embeds, bert_spans = bert.embed(text)

# q-labels
#   0: volume
#   1: speed
#   2: pitch_mean
#   3: pitch_fslope
#   4: pitch_lslope
#   5: pitch_rslope
words_regex = re.compile(r'\w+')
words = []
for match in words_regex.finditer(text):
    words.append((match.group(0), match.span(0)))

words, word_spans = zip(*words)

features = BatchPadded(
    text=text_encoded.unsqueeze(0),
    text_lengths=text_lengths,
    bert_embeds=torch.from_numpy(bert_embeds),
    bert_spans=torch.from_numpy(bert_spans.astype(int)).unsqueeze(0),
    bert_lengths=torch.tensor([len(bert_spans)], dtype=torch.long),
    spec=None,
    spec_lengths=None,
    wave=None,
    wave_lengths=None,
    qlabels=None,
    word_lengths=torch.tensor([len(words)], dtype=torch.long),
    word_spans=torch.tensor(list(word_spans), dtype=torch.long).unsqueeze(0),
)
net_g.enc_p.encoder.q_teacher_forcing = 0.0  # FIXME
*_, q_logits = net_g.enc_p.forward(features.text, features.text_lengths, features=features)
# TODO: make new targets for generator from predictions + self-made mask

# TODO: generate from there
y_hat, y_hat_mb, attn, mask, *_ = net_g.infer(x, x_lengths, batch, max_len=1000)
y_hat_lengths = mask.sum([1, 2]).long() * cfg.data.audio.hop_length

wave = y_hat[0, :, : y_hat_lengths[0]]
writewav(wave.detach().cpu().numpy(), cfg.data.audio.sampling_rate, 'gen.wav')
