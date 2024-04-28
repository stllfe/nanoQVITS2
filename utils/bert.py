"""Utilities for extracting features from a pretrained BERT-like encoders."""

from functools import cache

import torch

from numpy.typing import NDArray
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer


# this may be not the best model for the given task, check this later:
# https://github.com/avidale/encodechka
MODEL = 'cointegrated/rubert-tiny2'


@cache
def load() -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    model = AutoModel.from_pretrained(MODEL)
    model.eval()
    # todo: handle device
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    return model, tokenizer


@torch.inference_mode()
def embed(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
) -> tuple[NDArray, NDArray]:
    """Extracts token embeddings with the given model.

    Returns:
        A tuple of embeddings and token spans (T x 2 [start, end]) in the given texts:
    """

    data = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors='pt',
    )
    spans = data.pop('offset_mapping')[0]
    data.to(model.device)

    embeddings = model(**data)[0]
    embeddings = embeddings.detach().cpu().numpy()
    return embeddings, spans.numpy()
