"""Generate Infinity-2B/8B images with iterative last-scale schedules."""

import argparse
import hashlib
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("2b", "8b"), required=True)
    parser.add_argument(
        "--order",
        choices=("random", "sar", "sequential", "inside_out"),
        default="sar",
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vae-path", type=Path, required=True)
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=16,
                        help="Last-scale steps; the paper uses 16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=None, help="Defaults to 4.0 (2B) or 3.0 (8B)")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=900)
    parser.add_argument("--top-p", type=float, default=0.97)
    return parser


def _validate(args) -> list[str]:
    if not args.model_path.exists():
        raise FileNotFoundError(f"--model-path does not exist: {args.model_path}")
    for name in ("vae_path", "text_encoder", "prompts"):
        path = getattr(args, name)
        expected = path.is_dir() if name == "text_encoder" else path.is_file()
        if not expected:
            raise FileNotFoundError(f"--{name.replace('_', '-')} does not exist: {path}")
    if args.samples_per_prompt < 1:
        raise ValueError("--samples-per-prompt must be positive")
    if args.num_steps <= 0 or 4096 % args.num_steps:
        raise ValueError("--num-steps must be a positive divisor of 4096")
    prompts = [line.strip() for line in args.prompts.read_text(encoding="utf-8").splitlines()]
    prompts = [line for line in prompts if line and not line.startswith("#")]
    if not prompts:
        raise ValueError(f"no prompts found in {args.prompts}")
    return prompts


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    prompts = _validate(args)

    import cv2
    import torch
    from tools.run_infinity import (
        encode_prompt, load_tokenizer, load_transformer, load_visual_tokenizer,
    )
    from infinity.utils.dynamic_resolution import dynamic_resolution_h_w

    if not torch.cuda.is_available():
        raise RuntimeError("Full Infinity inference requires a CUDA GPU")
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    # These are the official Infinity-2B/8B 1M-resolution architecture settings.
    tokenizer, text_encoder = load_tokenizer(args.text_encoder)
    vae = load_visual_tokenizer(args.vae_path)
    model = load_transformer(args.model_path, vae, args.model)
    schedule = [
        (1, h, w) for _, h, w in dynamic_resolution_h_w[1.0]["1M"]["scales"]
    ]
    cfg = args.cfg if args.cfg is not None else (4.0 if args.model == "2b" else 3.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        (prompt_idx, prompt, sample_idx)
        for prompt_idx, prompt in enumerate(prompts)
        for sample_idx in range(args.samples_per_prompt)
    ][rank::world_size]
    for prompt_idx, prompt, sample_idx in jobs:
        seed = args.seed + prompt_idx * args.samples_per_prompt + sample_idx
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, _, images = model.autoregressive_infer_cfg_iterative(
                vae=vae, scale_schedule=schedule,
                label_B_or_BLT=encode_prompt(tokenizer, text_encoder, prompt),
                B=1, g_seed=seed, cfg_list=[cfg] * len(schedule),
                tau_list=[args.tau] * len(schedule), cfg_insertion_layer=[0],
                vae_type=32, ret_img=True, inference_mode=True,
                num_iterations=args.num_steps, order=args.order,
                top_k=args.top_k, top_p=args.top_p,
            )
        prompt_id = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
        output = args.output_dir / f"{prompt_idx:05d}_{prompt_id}_{sample_idx:02d}_seed{seed}.png"
        if not cv2.imwrite(str(output), images[0].cpu().numpy()):
            raise OSError(f"failed to write image: {output}")


if __name__ == "__main__":
    main()
