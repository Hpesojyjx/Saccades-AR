"""Combine each prompt's four Infinity samples into a DPG-Bench 2x2 grid."""

import argparse
import re
from collections import defaultdict
from pathlib import Path


NAME = re.compile(r"^(\d{5}_[0-9a-f]{12})_(\d{2})_seed-?\d+\.png$")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    from PIL import Image

    groups = defaultdict(list)
    for path in sorted(args.input_dir.glob("*.png")):
        match = NAME.match(path.name)
        if match:
            groups[match.group(1)].append((int(match.group(2)), path))
    if not groups:
        raise ValueError(f"no public Infinity result files found in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for prompt_id, samples in groups.items():
        samples.sort()
        if len(samples) != 4 or [index for index, _ in samples] != [0, 1, 2, 3]:
            raise ValueError(f"{prompt_id} requires samples 00, 01, 02, 03")
        images = []
        for _, path in samples:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        width, height = images[0].size
        if any(image.size != (width, height) for image in images):
            raise ValueError(f"image sizes differ for {prompt_id}")
        grid = Image.new("RGB", (2 * width, 2 * height))
        for index, image in enumerate(images):
            grid.paste(image, ((index % 2) * width, (index // 2) * height))
        grid.save(args.output_dir / f"{prompt_id}.png")


if __name__ == "__main__":
    main()
