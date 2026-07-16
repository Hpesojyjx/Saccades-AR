# VAR Experiments

This directory provides four-step VAR retraining and class-conditional inference with `original`, `random`, and `sar` generation orders.

Run all commands from the repository root.

## Installation

```bash
conda create -n sar-var python=3.10
conda activate sar-var
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements/var.txt
```

## Required inputs

Download the model initialization and VAE checkpoints:

- [VAR-d16](https://huggingface.co/FoundationVision/var/resolve/main/var_d16.pth)
- [VAR-d20](https://huggingface.co/FoundationVision/var/resolve/main/var_d20.pth)
- [VAE](https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth)

Arrange ImageNet-1K as a torchvision `ImageFolder`:

```text
/path/to/imagenet/
├── train/
│   ├── n01440764/
│   └── ...
└── val/
    ├── n01440764/
    └── ...
```

## Four-step retraining

### VAR-d16

```bash
NPROC_PER_NODE=8 bash var/scripts/train.sh \
  --depth 16 \
  --data-root /path/to/imagenet \
  --pretrained /path/to/var_d16.pth \
  --vae-checkpoint /path/to/vae_ch160v4096z32.pth \
  --output-dir ./checkpoints/var_d16_sar \
  --num-steps 4 \
  --bs 768 \
  --ep 250 \
  --seed YOUR_SEED
```

### VAR-d20

```bash
NPROC_PER_NODE=8 bash var/scripts/train.sh \
  --depth 20 \
  --data-root /path/to/imagenet \
  --pretrained /path/to/var_d20.pth \
  --vae-checkpoint /path/to/vae_ch160v4096z32.pth \
  --output-dir ./checkpoints/var_d20_sar \
  --num-steps 4 \
  --bs 768 \
  --ep 250 \
  --seed YOUR_SEED
```

The training command writes the latest checkpoint to `OUTPUT_DIR/ar-ckpt-last.pth`. Reusing the same output directory resumes from the latest compatible checkpoint.

## Inference

Random and SAR must use the same retrained checkpoint and sampling configuration. The examples below use VAR-d16; use `--depth 20` and the matching retrained checkpoint for VAR-d20.

### Published one-step reference

```bash
bash var/scripts/infer.sh \
  --depth 16 \
  --order original \
  --checkpoint /path/to/var_d16.pth \
  --vae-checkpoint /path/to/vae_ch160v4096z32.pth \
  --labels 0,1,2 \
  --samples-per-label 4 \
  --seed YOUR_SEED \
  --output-dir ./outputs/var_original_published
```

### Retrained one-step reference

```bash
bash var/scripts/infer.sh \
  --depth 16 \
  --order original \
  --checkpoint ./checkpoints/var_d16_sar/ar-ckpt-last.pth \
  --vae-checkpoint /path/to/vae_ch160v4096z32.pth \
  --labels 0,1,2 \
  --samples-per-label 4 \
  --seed YOUR_SEED \
  --output-dir ./outputs/var_original
```

### Random order

```bash
bash var/scripts/infer.sh \
  --depth 16 \
  --order random \
  --num-steps 4 \
  --checkpoint ./checkpoints/var_d16_sar/ar-ckpt-last.pth \
  --vae-checkpoint /path/to/vae_ch160v4096z32.pth \
  --labels 0,1,2 \
  --samples-per-label 4 \
  --seed YOUR_SEED \
  --output-dir ./outputs/var_random
```

### SAR order

```bash
bash var/scripts/infer.sh \
  --depth 16 \
  --order sar \
  --num-steps 4 \
  --checkpoint ./checkpoints/var_d16_sar/ar-ckpt-last.pth \
  --vae-checkpoint /path/to/vae_ch160v4096z32.pth \
  --labels 0,1,2 \
  --samples-per-label 4 \
  --seed YOUR_SEED \
  --output-dir ./outputs/var_sar
```

The default sampling settings are CFG 1.5, top-k 900, and top-p 0.96. Override them with `--cfg`, `--top-k`, and `--top-p`.

Each output is named:

```text
class<label>_sample<sample-index>_seed<seed>.png
```

## Full ImageNet generation

Generate a comma-separated label list and request the desired number of images per class:

```bash
LABELS=$(python -c 'print(",".join(map(str, range(1000))))')

for ORDER in original random sar; do
  bash var/scripts/infer.sh \
    --depth 16 \
    --order "${ORDER}" \
    --num-steps 4 \
    --checkpoint ./checkpoints/var_d16_sar/ar-ckpt-last.pth \
    --vae-checkpoint /path/to/vae_ch160v4096z32.pth \
    --labels "${LABELS}" \
    --samples-per-label 50 \
    --seed YOUR_SEED \
    --output-dir "./outputs/var_d16_${ORDER}_50k"
done
```

Generate the published one-step reference with the original checkpoint:

```bash
bash var/scripts/infer.sh \
  --depth 16 \
  --order original \
  --checkpoint /path/to/var_d16.pth \
  --vae-checkpoint /path/to/vae_ch160v4096z32.pth \
  --labels "${LABELS}" \
  --samples-per-label 50 \
  --seed YOUR_SEED \
  --output-dir ./outputs/var_d16_original_published_50k
```

For multi-GPU generation, prefix any inference command with `NPROC_PER_NODE=8`. Jobs are divided across ranks.

## FID evaluation

```bash
pip install -r requirements/evaluation.txt

python -m pytorch_fid \
  ./outputs/var_d16_sar_50k \
  /path/to/adm/imagenet256_stats.npz \
  --device cuda:0
```

Use the same labels, sample count, seed, CFG, top-k, and top-p settings when comparing Random and SAR.

## CLI reference

```bash
python var/train.py --help
python var/infer.py --help
bash var/scripts/train.sh --help
bash var/scripts/infer.sh --help
```
