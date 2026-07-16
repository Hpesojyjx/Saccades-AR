# Infinity Experiments

This directory provides Infinity-2B/8B text-to-image inference with `random`, `sar`, `sequential`, and `inside_out` generation orders. Iterative scheduling is applied to the final 4096-token scale; earlier scales use the original Infinity decoding path.

Run all commands from the repository root.

## Installation

```bash
conda create -n sar-infinity python=3.10
conda activate sar-infinity
pip install -r requirements/infinity.txt
```

GPU inference requires PyTorch 2.5.1 and a `flash-attn` build compatible with the installed CUDA toolkit and compiler.

## Required inputs

- [Infinity-2B checkpoint](https://huggingface.co/FoundationVision/infinity/blob/main/infinity_2b_reg.pth)
- [Infinity-8B sharded checkpoint directory](https://huggingface.co/FoundationVision/Infinity/tree/main/infinity_8b_weights)
- [Infinity regularized BSQ-VAE](https://huggingface.co/FoundationVision/Infinity/blob/main/infinity_vae_d32reg.pth)
- [FLAN-T5-XL](https://huggingface.co/google/flan-t5-xl)

`--model-path` accepts the Infinity-2B checkpoint file or the downloaded Infinity-8B sharded checkpoint directory.

Prompt files are UTF-8 text files with one prompt per line. Empty lines and lines beginning with `#` are ignored:

```text
a red fox sitting beside a white rabbit
four ceramic vases on a wooden table
```

## Generation orders

| `--order` | Position selection |
|---|---|
| `random` | Uniform random selection from uncommitted positions |
| `sar` | Smallest squared L2 distance to the pooled text-conditioning embedding |
| `sequential` | Lowest-index uncommitted positions |
| `inside_out` | Uncommitted positions nearest to the one-dimensional token midpoint |

## Infinity-2B inference

The default Infinity-2B CFG is 4.0.

### Random

```bash
bash infinity/scripts/infer.sh \
  --model 2b \
  --order random \
  --num-steps 16 \
  --model-path /path/to/infinity_2b_reg.pth \
  --vae-path /path/to/infinity_vae_d32reg.pth \
  --text-encoder /path/to/flan-t5-xl \
  --prompts examples/prompts.txt \
  --seed YOUR_SEED \
  --output-dir ./outputs/infinity_2b_random
```

### SAR

```bash
bash infinity/scripts/infer.sh \
  --model 2b \
  --order sar \
  --num-steps 16 \
  --model-path /path/to/infinity_2b_reg.pth \
  --vae-path /path/to/infinity_vae_d32reg.pth \
  --text-encoder /path/to/flan-t5-xl \
  --prompts examples/prompts.txt \
  --seed YOUR_SEED \
  --output-dir ./outputs/infinity_2b_sar
```

## Infinity-8B inference

The default Infinity-8B CFG is 3.0.

### Random

```bash
bash infinity/scripts/infer.sh \
  --model 8b \
  --order random \
  --num-steps 16 \
  --model-path /path/to/infinity_8b_weights \
  --vae-path /path/to/infinity_vae_d32reg.pth \
  --text-encoder /path/to/flan-t5-xl \
  --prompts examples/prompts.txt \
  --seed YOUR_SEED \
  --output-dir ./outputs/infinity_8b_random
```

### SAR

```bash
bash infinity/scripts/infer.sh \
  --model 8b \
  --order sar \
  --num-steps 16 \
  --model-path /path/to/infinity_8b_weights \
  --vae-path /path/to/infinity_vae_d32reg.pth \
  --text-encoder /path/to/flan-t5-xl \
  --prompts examples/prompts.txt \
  --seed YOUR_SEED \
  --output-dir ./outputs/infinity_8b_sar
```

## Infinity-8B deterministic-order ablation

Run Sequential and Inside-out with the same model, prompts, sampling parameters, and seed used for the Random and SAR runs:

```bash
for ORDER in sequential inside_out; do
  bash infinity/scripts/infer.sh \
    --model 8b \
    --order "${ORDER}" \
    --num-steps 16 \
    --cfg 3.0 \
    --tau 1.0 \
    --model-path /path/to/infinity_8b_weights \
    --vae-path /path/to/infinity_vae_d32reg.pth \
    --text-encoder /path/to/flan-t5-xl \
    --prompts /path/to/benchmark_prompts.txt \
    --samples-per-prompt 4 \
    --seed YOUR_SEED \
    --output-dir "./outputs/infinity_8b_${ORDER}"
done
```

## GenEval generation

Apply the Infinity prompt-rewriting protocol to the GenEval prompts, store one rewritten prompt per line in evaluator order, then generate four samples per prompt:

```bash
for ORDER in random sequential inside_out sar; do
  bash infinity/scripts/infer.sh \
    --model 8b \
    --order "${ORDER}" \
    --num-steps 16 \
    --cfg 3.0 \
    --tau 1.0 \
    --model-path /path/to/infinity_8b_weights \
    --vae-path /path/to/infinity_vae_d32reg.pth \
    --text-encoder /path/to/flan-t5-xl \
    --prompts /path/to/geneval_prompts.txt \
    --samples-per-prompt 4 \
    --seed YOUR_SEED \
    --output-dir "./outputs/geneval_infinity_8b_${ORDER}"
done
```

Run the official [GenEval evaluator](https://github.com/djghosh13/geneval) on the generated directory.

## DPG-Bench generation

Generate four samples per prompt:

```bash
for ORDER in random sequential inside_out sar; do
  bash infinity/scripts/infer.sh \
    --model 8b \
    --order "${ORDER}" \
    --num-steps 16 \
    --cfg 3.0 \
    --tau 1.0 \
    --model-path /path/to/infinity_8b_weights \
    --vae-path /path/to/infinity_vae_d32reg.pth \
    --text-encoder /path/to/flan-t5-xl \
    --prompts /path/to/dpg_bench_prompts.txt \
    --samples-per-prompt 4 \
    --seed YOUR_SEED \
    --output-dir "./outputs/dpg_infinity_8b_${ORDER}"

  python infinity/evaluation/make_2x2_grids.py \
    --input-dir "./outputs/dpg_infinity_8b_${ORDER}" \
    --output-dir "./outputs/dpg_infinity_8b_${ORDER}_grids"
done
```

Pass each grid directory to the official DPG-Bench evaluator.

## MJHQ-30K generation and FID

```bash
bash infinity/scripts/infer.sh \
  --model 8b \
  --order sar \
  --num-steps 16 \
  --cfg 3.0 \
  --tau 1.0 \
  --model-path /path/to/infinity_8b_weights \
  --vae-path /path/to/infinity_vae_d32reg.pth \
  --text-encoder /path/to/flan-t5-xl \
  --prompts /path/to/mjhq30k_prompts.txt \
  --samples-per-prompt 1 \
  --seed YOUR_SEED \
  --output-dir ./outputs/mjhq30k_infinity_8b_sar

pip install -r requirements/evaluation.txt
python -m pytorch_fid \
  ./outputs/mjhq30k_infinity_8b_sar \
  /path/to/mjhq30k/reference/images \
  --device cuda:0
```

## Multi-GPU inference

Set `NPROC_PER_NODE` to divide prompt/sample jobs across ranks:

```bash
NPROC_PER_NODE=8 bash infinity/scripts/infer.sh \
  --model 8b \
  --order sar \
  --num-steps 16 \
  --model-path /path/to/infinity_8b_weights \
  --vae-path /path/to/infinity_vae_d32reg.pth \
  --text-encoder /path/to/flan-t5-xl \
  --prompts /path/to/prompts.txt \
  --seed YOUR_SEED \
  --output-dir ./outputs/infinity_8b_sar
```

Each generated image is named:

```text
<prompt-index>_<prompt-hash>_<sample-index>_seed<seed>.png
```

## CLI reference

```bash
python infinity/infer.py --help
python infinity/evaluation/make_2x2_grids.py --help
bash infinity/scripts/infer.sh --help
```
