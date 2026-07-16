"""Public VAR inference CLI for the original, random, and SAR schedules."""

import argparse
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, choices=(16, 20), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--order", choices=("original", "random", "sar"), default="sar")
    parser.add_argument("--num-steps", type=int, default=4,
                        help="Last-scale steps; the paper uses 4")
    parser.add_argument("--labels", required=True, help="Comma-separated ImageNet class IDs")
    parser.add_argument("--samples-per-label", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=1.5)
    parser.add_argument("--top-k", type=int, default=900)
    parser.add_argument("--top-p", type=float, default=0.96)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def _extract_var_state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("VAR checkpoint must contain a state-dict mapping")
    if isinstance(checkpoint.get("trainer"), dict):
        checkpoint = checkpoint["trainer"].get("var_wo_ddp", checkpoint["trainer"])
    for key in ("var_wo_ddp", "state_dict", "model", "module"):
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get(key), dict):
            checkpoint = checkpoint[key]
    prefixes = ("module.", "_orig_mod.")
    cleaned = {}
    for key, value in checkpoint.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    return cleaned


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    for path_arg in (args.checkpoint, args.vae_checkpoint):
        if not path_arg.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {path_arg}")
    labels = [int(value) for value in args.labels.split(",")]
    if not labels or any(not 0 <= value < 1000 for value in labels):
        raise ValueError("--labels must contain ImageNet class IDs in [0, 999]")
    if args.samples_per_label < 1:
        raise ValueError("--samples-per-label must be positive")
    if args.num_steps <= 0 or 256 % args.num_steps:
        raise ValueError("--num-steps must be a positive divisor of 256")

    import torch
    from PIL import Image
    from models import build_vae_var

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu only for debugging")
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if args.device == "cuda" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    vae, model = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        num_classes=1000, depth=args.depth, shared_aln=False,
        attn_l2_norm=True, flash_if_available=device.type == "cuda",
        fused_if_available=device.type == "cuda", num_gen_steps=args.num_steps,
    )
    vae.load_state_dict(torch.load(args.vae_checkpoint, map_location="cpu"), strict=True)
    model.load_state_dict(
        _extract_var_state(torch.load(args.checkpoint, map_location="cpu")), strict=True
    )
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        (label, sample_idx)
        for label in labels
        for sample_idx in range(args.samples_per_label)
    ][rank::world_size]
    for label, sample_idx in jobs:
        seed = args.seed + label * args.samples_per_label + sample_idx
        if args.order == "original":
            image = model.autoregressive_infer_cfg(
                B=1, label_B=label, g_seed=seed, cfg=args.cfg,
                top_k=args.top_k, top_p=args.top_p,
            )
        else:
            image = model.autoregressive_infer_cfg_iterative(
                B=1, label_B=label, g_seed=seed, cfg=args.cfg,
                top_k=args.top_k, top_p=args.top_p,
                num_iterations=args.num_steps, order=args.order,
            )
        array = image[0].detach().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
        Image.fromarray(array).save(
            args.output_dir / f"class{label:04d}_sample{sample_idx:04d}_seed{seed}.png"
        )


if __name__ == "__main__":
    main()
