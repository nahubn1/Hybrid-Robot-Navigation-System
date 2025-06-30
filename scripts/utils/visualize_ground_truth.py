#!/usr/bin/env python3
"""Visualize generated ground truth outputs."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main(sample_prefix: str):
    base = Path(sample_prefix)
    indices = np.load(base.with_suffix('.indices.npy'))
    mask = np.load(base.with_suffix('.mask.npy'))
    heat = np.load(base.with_suffix('.heat.npy'))

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(indices, origin='lower')
    ax[0].set_title('Indices')
    ax[1].imshow(mask, cmap='gray', origin='lower')
    ax[1].set_title('Mask')
    ax[2].imshow(heat, cmap='hot', origin='lower')
    ax[2].set_title('Heatmap')
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Visualize ground truth outputs')
    p.add_argument('prefix', type=str, help='Path prefix of sample (without extension)')
    args = p.parse_args()
    main(args.prefix)
