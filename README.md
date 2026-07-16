# Saccade Autoregressive Generation

Official experiment code for **Context-Aware Autoregressive Image Generation for Emerging Reasoning Properties**.

Saccade Autoregressive Generation (SAR) changes the order in which an autoregressive image model commits visual tokens. At each iterative decoding step, SAR ranks uncommitted positions using the distance between their current hidden states and the model's conditioning embedding, then commits the most context-relevant positions first. This repository provides SAR implementations for VAR and Infinity.

## Repository layout

```text
SAR-public-release/
├── var/                       # VAR training and class-conditional inference
├── infinity/                  # Infinity-2B/8B text-to-image inference
├── requirements/              # Separate VAR, Infinity, and evaluation environments
├── examples/prompts.txt       # Example text prompts
└── tests/                     # CPU scheduling and public CLI tests
```

Detailed usage:

- [VAR experiments](var/README.md)
- [Infinity experiments](infinity/README.md)

Run all commands below from the repository root.

## Installation

VAR and Infinity require different PyTorch versions. Install them in separate Python 3.10 environments.

### VAR environment

```bash
conda create -n sar-var python=3.10
conda activate sar-var
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements/var.txt
```

### Infinity environment

```bash
conda create -n sar-infinity python=3.10
conda activate sar-infinity
pip install -r requirements/infinity.txt
```

Infinity GPU inference also requires a `flash-attn` build compatible with the installed CUDA toolkit and PyTorch version.

## Required resources

### VAR

- [VAR-d16 checkpoint](https://huggingface.co/FoundationVision/var/resolve/main/var_d16.pth)
- [VAR-d20 checkpoint](https://huggingface.co/FoundationVision/var/resolve/main/var_d20.pth)
- [VAR VAE checkpoint](https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth)
- ImageNet-1K arranged as a torchvision `ImageFolder`

### Infinity

- [Infinity-2B checkpoint](https://huggingface.co/FoundationVision/infinity/blob/main/infinity_2b_reg.pth)
- [Infinity-8B sharded checkpoint](https://huggingface.co/FoundationVision/Infinity/tree/main/infinity_8b_weights)
- [Infinity regularized BSQ-VAE](https://huggingface.co/FoundationVision/Infinity/blob/main/infinity_vae_d32reg.pth)
- [FLAN-T5-XL](https://huggingface.co/google/flan-t5-xl)
- A UTF-8 prompt file with one prompt per line

## Experiment commands

### Train a four-step VAR checkpoint

```bash
for DEPTH in 16 20; do
  NPROC_PER_NODE=8 bash var/scripts/train.sh \
    --depth "${DEPTH}" \
    --data-root /path/to/imagenet \
    --pretrained "/path/to/var_d${DEPTH}.pth" \
    --vae-checkpoint /path/to/vae_ch160v4096z32.pth \
    --output-dir "./checkpoints/var_d${DEPTH}_sar" \
    --num-steps 4 \
    --bs 768 \
    --ep 250 \
    --seed YOUR_SEED
done
```

### Run VAR one-step, Random, and SAR inference

```bash
# One-step reference using the retrained checkpoint
bash var/scripts/infer.sh \
  --depth 16 \
  --order original \
  --checkpoint ./checkpoints/var_d16_sar/ar-ckpt-last.pth \
  --vae-checkpoint /path/to/vae_ch160v4096z32.pth \
  --labels 0,1,2 \
  --samples-per-label 4 \
  --seed YOUR_SEED \
  --output-dir ./outputs/var_original

# Matched Random and SAR inference
for ORDER in random sar; do
  bash var/scripts/infer.sh \
    --depth 16 \
    --order "${ORDER}" \
    --num-steps 4 \
    --checkpoint ./checkpoints/var_d16_sar/ar-ckpt-last.pth \
    --vae-checkpoint /path/to/vae_ch160v4096z32.pth \
    --labels 0,1,2 \
    --samples-per-label 4 \
    --seed YOUR_SEED \
    --output-dir "./outputs/var_${ORDER}"
done
```

### Run Infinity-2B Random and SAR inference

```bash
for ORDER in random sar; do
  bash infinity/scripts/infer.sh \
    --model 2b \
    --order "${ORDER}" \
    --num-steps 16 \
    --model-path /path/to/infinity_2b_reg.pth \
    --vae-path /path/to/infinity_vae_d32reg.pth \
    --text-encoder /path/to/flan-t5-xl \
    --prompts examples/prompts.txt \
    --seed YOUR_SEED \
    --output-dir "./outputs/infinity_2b_${ORDER}"
done
```

### Run Infinity-8B Random and SAR inference

```bash
for ORDER in random sar; do
  bash infinity/scripts/infer.sh \
    --model 8b \
    --order "${ORDER}" \
    --num-steps 16 \
    --model-path /path/to/infinity_8b_weights \
    --vae-path /path/to/infinity_vae_d32reg.pth \
    --text-encoder /path/to/flan-t5-xl \
    --prompts examples/prompts.txt \
    --seed YOUR_SEED \
    --output-dir "./outputs/infinity_8b_${ORDER}"
done
```

### Run the Infinity-8B deterministic-order ablation

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

### Generate GenEval images

Prepare the Infinity-rewritten GenEval prompts as a UTF-8 file with one prompt per line, preserving evaluator order.

```bash
for ORDER in random sequential inside_out sar; do
  bash infinity/scripts/infer.sh \
    --model 8b \
    --order "${ORDER}" \
    --num-steps 16 \
    --model-path /path/to/infinity_8b_weights \
    --vae-path /path/to/infinity_vae_d32reg.pth \
    --text-encoder /path/to/flan-t5-xl \
    --prompts /path/to/geneval_prompts.txt \
    --samples-per-prompt 4 \
    --seed YOUR_SEED \
    --output-dir "./outputs/geneval_infinity_8b_${ORDER}"
done
```

### Generate DPG-Bench images and 2x2 grids

```bash
for ORDER in random sequential inside_out sar; do
  bash infinity/scripts/infer.sh \
    --model 8b \
    --order "${ORDER}" \
    --num-steps 16 \
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

Use the official GenEval and DPG-Bench evaluators to score the generated images.

### Generate MJHQ-30K images and compute FID

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

### Compute ImageNet FID for VAR

```bash
pip install -r requirements/evaluation.txt
python -m pytorch_fid \
  /path/to/generated/images \
  /path/to/adm/imagenet256_stats.npz \
  --device cuda:0
```

See the [VAR guide](var/README.md) and [Infinity guide](infinity/README.md) for full-dataset generation, output naming, distributed execution, and benchmark-specific commands.

## Citation

```bibtex
@article{ying2026contextaware,
  title   = {Context-Aware Autoregressive Image Generation for Emerging Reasoning Properties},
  author  = {Ying, Jixuan and Liu, Haoyu and Yang, Timing and Zhu, Tingyu and Neskovic, Predrag and Bhattad, Anand and Xie, Cihang and Zheng, Zeyu and Yuille, Alan and Wang, Feng},
  year    = {2026}
}
```

## License

The copied VAR and Infinity code is released under the FoundationVision MIT License. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for license and attribution details.
