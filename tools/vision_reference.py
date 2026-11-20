"""Gemma 4 vision features from mlx-vlm, for Syrinx's gemma4_vision test to check against.

Runs mlx-vlm's Gemma 4 VisionModel in float32 on the test image the Rust test makes
(`test_image` in gemma4_vision.rs: same formula) and writes the standardized soft tokens
as little-endian f32s, then the final ones after `embed_vision` (which in the MLX 8-bit copy
is quantized, so those are only a loose check).

    ~/.venvs/mlxbench-080/bin/python -I tools/vision_reference.py \
        models/mlx/gemma-4-26B-A4B-it-MLX-8bit 144 192 /tmp/vision-ref.f32
"""
import json
import math
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm.models.gemma4.config import VisionConfig
from mlx_vlm.models.gemma4.vision import VisionModel


def test_image(h, w):
    img = np.zeros((h, w, 3), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            for c in range(3):
                img[y, x, c] = 0.5 + 0.5 * math.sin(x * 0.05 + c * 1.3) * math.cos(y * 0.07 - c * 0.4)
    return img


def main():
    folder, h, w, out = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    cfg = json.loads((folder / "config.json").read_text())["vision_config"]
    model = VisionModel(VisionConfig.from_dict(cfg))
    index = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
    weights = {}
    for shard in sorted(set(index.values())):
        for k, v in mx.load(str(folder / shard)).items():
            if k.startswith("vision_tower."):
                weights[k[len("vision_tower."):]] = v.astype(mx.float32)
    model.load_weights(list(weights.items()))
    img = test_image(h, w)
    pixels = mx.array(img.transpose(2, 0, 1)[None])  # B, C, H, W in 0..1
    feats = model(pixels)[0]
    mx.eval(feats)
    a = np.array(feats, dtype=np.float32)
    a.tofile(out)
    print(f"{a.shape[0]} soft tokens x {a.shape[1]} -> {out}; mean |x| {np.abs(a).mean():.4f}")


if __name__ == "__main__":
    main()
