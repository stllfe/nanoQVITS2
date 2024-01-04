"""Utilities for extracting features from a pretrained BERT-like encoders."""

import torch

from numpy.typing import NDArray
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)


MODEL = 'cointegrated/rubert-tiny2'


def load() -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    model = AutoModel.from_pretrained(MODEL)
    model.eval()
    # todo: handle device
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    print(f'Loaded {MODEL}')
    return model, tokenizer


@torch.inference_mode()
def embed(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
) -> tuple[NDArray, NDArray]:
    """Extracts token embeddings with the given model.

    Returns:
        A tuple of embeddings and token offsets (T x 2 [start, end]) in the given texts:
    """

    data = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors='pt',
    )
    offsets = data.pop('offset_mapping')[0]
    data.to(model.device)

    embeddings = model(**data)[0]
    embeddings = embeddings.detach().cpu().numpy()
    return embeddings, offsets.numpy()
