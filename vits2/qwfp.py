import math

import numpy as np
import torch

from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from utils.data import BatchPadded
from vits2 import commons
from vits2.attentions import Encoder


# TODO: implement these
# - [ ] Начисто реализовать QuantizedWordFeaturesPredictor
# 	- [ ] Внести код QuantizedWordFeaturesPredictor в TextEncoder, откатить модификацию общего для нескольких слоёв класса Encoder
# 	- [ ] Выделить отдельные функции, которые выравнивают word-token и token-char признаки и покрыть тестами
# 		- [ ] Заставить работать на батчах, проверить корректность вычислений
# 		- [ ] Проверить, что interspersed ничего не ломает
# 	- [ ] Сделать pack/unpack для последовательностей при подаче в RNN
# 	- [ ] Упростить определение архитектуры QWFP, сделать простые блоки со свёртками вместо вот этого ужаса сейчас: перейти на те же доступные resblocks?


def QWFPBlock(in_dim: int, out_dim: int, dropout: float = 0.0, bias: bool = True) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_dim, out_dim, kernel_size=(3,), padding=1, bias=bias),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
        nn.Dropout(dropout) if dropout else nn.Identity(),
    )


class QuantizedWordFeaturesPredictor(torch.nn.Module):
    def __init__(
        self,
        bert_emb_dim: int,
        text_emb_dim: int,
        n_feats: int = 6,
        q_dim: int = 5 + 1,  # additional for empty (pad) label
        p_dropout: float = 0.2,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.bert_emb_dim = bert_emb_dim
        self.n_features = n_feats
        self.q_dim = q_dim
        self.text_emb_dim = text_emb_dim
        self.hidden_dim = text_emb_dim
        initial_channel = bert_emb_dim + self.hidden_dim
        self.pool = nn.Sequential(
            QWFPBlock(initial_channel, self.hidden_dim, dropout=p_dropout, bias=bias),
            *(
                QWFPBlock(self.hidden_dim, self.hidden_dim, dropout=p_dropout, bias=bias)
                for _ in range(3)
            ),
        )
        self.token_rnn = torch.nn.GRU(
            self.hidden_dim,
            self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            bias=bias,
        )
        # TODO: instead, we can make a big single matrix and split it to n_features (just like QKV for attention)
        # just to pretend we are cool kids
        heads, embs = [], []
        for _ in range(self.n_features):
            head = torch.nn.Linear(self.hidden_dim * 2, q_dim)
            emb = nn.Embedding(q_dim, self.hidden_dim, padding_idx=0)
            # need to think more about initialization scheme for embeddings
            # since we sum them up, they should have some upper bound for summation, adjusted by the number of heads
            # is this enough?
            nn.init.normal_(emb.weight, mean=0, std=self.n_features**-0.5)
            heads.append(head)
            embs.append(emb)
        self.heads = nn.ModuleList(heads)
        self.embs = nn.ModuleList(embs)

    def embed(self, labels: torch.Tensor) -> torch.Tensor:
        embeds = []
        for j in range(self.n_features):
            embeds.append(self.embs[j](labels[:, :, j, :]))
        # [B, token_len, n_features, emb_dim]
        embeds = torch.cat(embeds, dim=2)
        return embeds

    def forward(
        self,
        x_cond: Tensor,
        x_mask: Tensor,
        bert_embeds: Tensor,
        bert_lengths: Tensor,
        bert_spans: Tensor,
    ) -> Tensor:
        # [B, text_hidden_channels, text_len] -> [B, text_len, text_hidden_channels]
        x_cond = x_cond * x_mask
        x = x_cond.permute(0, 2, 1)
        B, T = x.size(0), x.size(1)

        char_embeds = torch.zeros(
            (B, bert_embeds.shape[1], self.text_emb_dim),
            dtype=torch.float,
            device=x.device,
        )
        for i in range(B):
            # FIXME: need to figure out how to handle interspersed texts!

            # version 1 (current): we grab just a last char hidden from the text encoder
            # however we may loose some valuable info form the intersperse chars!
            # my hope though is that biRNN will utilize any valuable info from them if applies

            # hence, version 2 idea: maybe use the sum (or mean) of the char + its left-side intersperse? (maybe even right-side?)

            # as per authors we may select the last char hidden (of each token) from the text encoder outputs directly
            # but the problem with not taking intersperse neighbors into account still kinda applies then...

            L = bert_lengths[i].item()
            char_emb = torch.index_select(
                # FIXME: works only for interspersed version
                x[i][1::2],
                dim=0,
                index=torch.clamp(bert_spans[i, :L, 1] - 1, max=x[i][1::2].size(0) - 1),
            )
            # adjust to bert length
            char_embeds[i, :L] = char_emb

        x = torch.cat((bert_embeds, char_embeds), dim=2)
        # [B, token_len, bert_dim + text_hidden_dim]

        x = self.pool(x.transpose(1, 2))
        x = x.transpose(1, 2)

        # TODO: do we need it anyways?
        packed_x = pack_padded_sequence(
            x,
            bert_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_latents, _ = self.token_rnn(packed_x)  # [B, token_len, inner_hidden_dim * 2]
        latents, _ = pad_packed_sequence(packed_latents)
        latents = latents.permute(1, 0, 2)

        logits = []
        for j in range(self.n_features):
            logits.append(self.heads[j](latents).unsqueeze(3))

        # [B, token_len, q_dim, n_features]
        logits = torch.cat(logits, dim=3)
        return logits


class TextEncoderWithQWFP(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float = 0.1,
        gin_channels: int = 0,
        q_teacher_forcing: float = 1.0,
        q_condition_layer: int | None = None,
    ) -> None:
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels
        self.q_teacher_forcing = q_teacher_forcing
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.encoder = QWFPEncoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=self.gin_channels,
            q_teacher_forcing=self.q_teacher_forcing,
            q_condition_layer=q_condition_layer,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None, features: BatchPadded | None = None) -> torch.Tensor:
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x, q_loss = self.encoder(x * x_mask, x_mask, g=g, features=features)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask, q_loss


class QWFPEncoder(Encoder):
    def __init__(
        self, *args, q_teacher_forcing: float = 1, q_condition_layer: int | None = None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.q_teacher_forcing = q_teacher_forcing
        self.q_condition_layer = (
            self.n_layers - 1 if q_condition_layer is None else q_condition_layer
        )
        self.qwfp = QuantizedWordFeaturesPredictor(
            bert_emb_dim=312, text_emb_dim=self.hidden_channels, p_dropout=0.1
        )

    def compute_q(self, x: Tensor, x_mask: Tensor, features: BatchPadded) -> tuple[Tensor, Tensor]:
        # TODO: wrap it into aligning helper function
        # q + x -> continue from the third layer in transformer (actually move it to a text_encoder)
        # maybe we could just shift the spans correctly? the problem is that we don't wanna do anything for the interspersed embedding as of right now...

        q_logits = self.qwfp(
            # TODO: we currently don't propagate gradients from QWFP to TextEncoder (but maybe we should?)
            x.detach(),
            x_mask,
            features.bert_embeds,
            features.bert_lengths,
            features.bert_spans,
        )

        # we train this e2e, so using teacher forcing
        # I suspect it may result in slower convergence for both models, but maybe it becomes more robust?
        # under q_teacher_forcing = 1 it's exactly the original way of training
        if np.random.uniform(0, 1) <= self.q_teacher_forcing:
            q_labels, _ = get_token_level_q_targets(features)
        else:
            # [B, token_len, q_dim, n_features] -> [B, token_len, n_features]
            q_labels = q_logits.argmax(dim=2)

        # labels should be of [B, token_len, n_features, 1] here
        q_embeds = self.qwfp.embed(q_labels.unsqueeze(-1))

        # TODO: adapt embedding shapes so that we can add them directly to x_cond vectors
        # looks like it would be easier to fuse QWFP with the TextEncoder itself,
        # since we have to add it BEFORE the third layer
        # TODO instead of sum we can add a concat+proj layer -> weighted sum
        q_add = q_embeds.sum(dim=2).permute(0, 2, 1)

        # upsample q embeddings back to char level and add to x_cond
        interspersed_indices = torch.arange((x.size(2) - 1) // 2, device=x.device)
        bert_spans = features.bert_spans.unsqueeze(2)
        # fmt: off
        alignment = (
            (bert_spans[:, :, :, 1] > interspersed_indices) &
            (interspersed_indices >= bert_spans[:, :, :, 0])
        )
        # fmt: on
        alignment = alignment.int()
        tokens_mask = ~alignment.eq(0).all(dim=1)

        xq = torch.zeros_like(x)
        indices = alignment.argmax(dim=1)

        # FIXME: again, works on interspersed only
        xq[:, :, 1::2] = torch.where(
            tokens_mask.unsqueeze(1),
            q_add.take_along_dim(indices.unsqueeze(1), dim=2),
            xq[:, :, 1::2],
        )
        return xq, q_logits

    def forward(
        self, x, x_mask, g=None, features: BatchPadded | None = None
    ) -> tuple[Tensor, Tensor]:
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            # TODO: add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            # https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/models/vits/modeling_vits.py#L1187
            if i == self.q_condition_layer:
                if g is not None:
                    g = self.spk_emb_linear(g.transpose(1, 2))
                    g = g.transpose(1, 2)
                    x = x + g
                    x = x * x_mask
                if features is not None:
                    q, q_logits = self.compute_q(x, x_mask, features)
                    x = x + q
                    x = x * x_mask
                else:
                    # TODO: else compute it here ?
                    pass
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x, q_logits


def get_token_level_q_targets(features: BatchPadded) -> tuple[Tensor, Tensor]:
    # FIXME: padding for spans messes with the alignment
    bert_spans = features.bert_spans.unsqueeze(2)
    word_spans = features.word_spans.unsqueeze(1)
    # fmt: off
    alignment = (
        (bert_spans[:, :, :, 0] >= word_spans[:, :, :, 0]) &
        (bert_spans[:, :, :, 1] <= word_spans[:, :, :, 1])
    )
    # fmt: on
    alignment = alignment.int()
    words_mask = ~alignment.eq(0).all(dim=2)
    indices = torch.argmax(alignment, dim=2)
    gts = features.qlabels.take_along_dim(indices.unsqueeze(2), dim=1)
    q = torch.zeros_like(gts)
    q_target = torch.where(words_mask.unsqueeze(2), gts, q)
    return q_target, words_mask


def compute_q_loss(
    logits: Tensor,
    features: BatchPadded,
    n_features: int = 6,
    q_dim: int = 6,  # TODO: remove hard-coded dimensions
) -> Tensor:
    target, _ = get_token_level_q_targets(features)
    mask = commons.sequence_mask(features.bert_lengths, max_length=features.bert_spans.size(1))
    target.masked_fill_(mask.unsqueeze(-1), 0)
    if logits.size(1) < target.size(1):
        assert logits.size(0) == 1, 'Should be a single batch element then!'
        logits = F.pad(
            logits,
            (0, 0, 0, 0, 0, target.size(1) - logits.size(1)),
            value=0,
            mode='constant',
        )
    return F.cross_entropy(
        logits.view(-1, n_features, q_dim),
        target.view(-1, n_features).long(),  # FIXME: prepare at batch-time?
        ignore_index=0,
    )
