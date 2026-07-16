"""Minimal Infinity-2B/8B loading helpers used by the public SAR CLI."""

from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel, T5TokenizerFast

from infinity.models.infinity import Infinity


def encode_prompt(text_tokenizer, text_encoder, prompt):
    """Encode one prompt in Infinity's compact cross-attention format."""
    tokens = text_tokenizer(
        text=[prompt], max_length=512, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    device = next(text_encoder.parameters()).device
    input_ids = tokens.input_ids.to(device, non_blocking=True)
    mask = tokens.attention_mask.to(device, non_blocking=True)
    text_features = text_encoder(
        input_ids=input_ids, attention_mask=mask
    )["last_hidden_state"].float()
    lens: List[int] = mask.sum(dim=-1).tolist()
    cu_seqlens_k = F.pad(
        mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0)
    )
    compact = torch.cat(
        [features[:length] for length, features in zip(lens, text_features)], dim=0
    )
    return compact, lens, cu_seqlens_k, max(lens)


def load_tokenizer(t5_path, device="cuda"):
    """Load the frozen FLAN-T5-XL encoder used by the paper."""
    tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(
        t5_path, revision=None, legacy=True
    )
    tokenizer.model_max_length = 512
    encoder: T5EncoderModel = T5EncoderModel.from_pretrained(
        t5_path, torch_dtype=torch.float16
    )
    encoder.to(device).eval().requires_grad_(False)
    return tokenizer, encoder


def load_visual_tokenizer(vae_path, device="cuda", apply_spatial_patchify=False):
    """Load the public Infinity d32 regularized BSQ-VAE."""
    from infinity.models.bsq_vae.vae import vae_model

    if apply_spatial_patchify:
        patch_size = 8
        encoder_ch_mult = [1, 2, 4, 4]
        decoder_ch_mult = [1, 2, 4, 4]
    else:
        patch_size = 16
        encoder_ch_mult = [1, 2, 4, 4, 4]
        decoder_ch_mult = [1, 2, 4, 4, 4]
    return vae_model(
        str(vae_path), "dynamic", 32, 2**32, patch_size=patch_size,
        encoder_ch_mult=encoder_ch_mult, decoder_ch_mult=decoder_ch_mult,
        test_mode=True,
    ).to(device)


def _extract_model_state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Infinity checkpoint must contain a state-dict mapping")
    for key in ("state_dict", "model", "module"):
        if isinstance(checkpoint.get(key), dict):
            checkpoint = checkpoint[key]
    return checkpoint


def _load_model_state(model, model_path, device):
    """Load an Infinity checkpoint from a file or a sharded directory."""
    model_path = Path(model_path)
    if model_path.is_dir():
        from transformers.modeling_utils import load_sharded_checkpoint

        return load_sharded_checkpoint(model, model_path, strict=False)
    state = _extract_model_state(torch.load(model_path, map_location=device))
    return model.load_state_dict(state)


def load_transformer(model_path, vae, model_size, device="cuda", bf16=True):
    """Construct and load the public Infinity-2B or Infinity-8B model."""
    if model_size == "2b":
        architecture = dict(
            depth=32, embed_dim=2048, num_heads=16,
            drop_path_rate=0.1, mlp_ratio=4, block_chunks=8,
        )
    elif model_size == "8b":
        architecture = dict(
            depth=40, embed_dim=3584, num_heads=28,
            drop_path_rate=0.1, mlp_ratio=4, block_chunks=8,
        )
    else:
        raise ValueError(f"model_size must be '2b' or '8b', got {model_size!r}")

    with torch.no_grad():
        model = Infinity(
            vae_local=vae, text_channels=2048, text_maxlen=512,
            shared_aln=True, raw_scale_schedule=None,
            checkpointing="full-block", customized_flash_attn=False,
            fused_norm=True, pad_to_multiplier=128, use_flex_attn=False,
            add_lvl_embeding_only_first_block=False, use_bit_label=True,
            rope2d_each_sa_layer=True, rope2d_normalized_by_hw=2,
            pn="1M", apply_spatial_patchify=False, inference_mode=True,
            train_h_div_w_list=[1.0], **architecture,
        ).to(device)
        if bf16:
            for block in model.unregistered_blocks:
                block.bfloat16()
        incompatible = _load_model_state(model, model_path, device)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(f"Infinity checkpoint compatibility: {incompatible}")
        model.eval().requires_grad_(False)
        model.rng = torch.Generator(device=device)
    return model
