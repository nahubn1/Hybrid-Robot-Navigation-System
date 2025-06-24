#!/usr/bin/env python3
"""Orchestrate full dataset generation pipeline."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

import pybullet as p
import pybullet_data

SRC_PATH = Path(__file__).resolve().parents[2] / 'src'
import sys
sys.path.append(str(SRC_PATH))

from data_generation.pybullet_scene_generator import (
    create_cluttered_scene,
    create_room_scene,
    create_maze_scene,
)
from data_generation.raycast_utils import generate_occupancy_grid
from data_generation.ground_truth_heatmaps import (
    PRMConfig,
    HeatmapConfig,
    generate_heatmap,
)

ARCTYPES = ['clutter', 'room', 'maze']


def load_config(path: Path) -> Dict:
    with open(path, 'r') as f:
        if path.suffix in {'.yaml', '.yml'}:
            return yaml.safe_load(f)
        return json.load(f)


def generate_envs(archetype: str, count: int, out_dir: Path, cfg: Dict, seed: int) -> None:
    mode = p.DIRECT
    p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    asset_dir = Path(__file__).resolve().parents[2] / 'assets'
    cube_urdf = str(asset_dir / 'cube.urdf')
    wall_urdf = str(asset_dir / 'wall.urdf')

    for idx in range(count):
        scene_seed = seed + idx
        random.seed(scene_seed)
        np.random.seed(scene_seed)
        p.resetSimulation()
        p.loadURDF('plane.urdf')
        if archetype == 'clutter':
            create_cluttered_scene(cfg['obstacle_count'], cfg['area_size'], urdf_path=cube_urdf, seed=scene_seed)
        elif archetype == 'room':
            create_room_scene(cfg['num_rooms'], cfg['area_size'], urdf_path=wall_urdf, seed=scene_seed)
        else:
            create_maze_scene(cfg['maze_size'], cfg['passage_width'], urdf_path=wall_urdf, seed=scene_seed)

        grid = generate_occupancy_grid(cfg['area_size'], cfg['resolution'], cfg['raycast_height'])
        base = f"env_{archetype}_{idx:04d}"
        np.save(out_dir / f"{base}.npy", grid)
        meta = {
            'seed': scene_seed,
            'archetype': archetype,
            'index': idx,
            'area_size': cfg['area_size'],
            'resolution': cfg['resolution'],
        }
        with open(out_dir / f"{base}.json", 'w') as f:
            json.dump(meta, f, indent=2)

    p.disconnect()


def generate_ground_truth(env_dir: Path, cfg: Dict) -> None:
    gt_dir = env_dir / 'gt'
    gt_dir.mkdir(parents=True, exist_ok=True)
    prm_cfg = PRMConfig(num_samples=cfg['samples'], radius=cfg['radius'])
    hm_cfg = HeatmapConfig(sigma=cfg['sigma'], top_k=cfg['top_k'])
    method = cfg.get('method', 'betweenness')

    for grid_path in sorted(env_dir.glob('env_*.npy')):
        grid = np.load(grid_path)
        heatmap = generate_heatmap(grid, prm_cfg, hm_cfg, method)
        out_file = gt_dir / grid_path.name.replace('env_', 'gt_')
        np.save(out_file, heatmap)


def collect_pairs(raw_root: Path) -> Dict[str, List[Tuple[Path, Path]]]:
    pairs: Dict[str, List[Tuple[Path, Path]]] = {}
    for arch_dir in raw_root.iterdir():
        if not arch_dir.is_dir():
            continue
        for img in arch_dir.glob('env_*.npy'):
            label = arch_dir / 'gt' / img.name.replace('env_', 'gt_')
            if not label.exists():
                continue
            meta_path = img.with_suffix('.json')
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                archetype = meta.get('archetype', arch_dir.name)
            else:
                archetype = arch_dir.name
            pairs.setdefault(archetype, []).append((img, label))
    return pairs


def split_pairs(pairs_by_arch: Dict[str, List[Tuple[Path, Path]]], ratios: Dict[str, float], seed: int) -> Dict[str, List[Tuple[Path, Path]]]:
    splits = {k: [] for k in ['train', 'val', 'test']}
    rng = random.Random(seed)
    for arch, pairs in pairs_by_arch.items():
        rng.shuffle(pairs)
        n = len(pairs)
        n_train = int(n * ratios.get('train', 0.8))
        n_val = int(n * ratios.get('val', 0.1))
        n_test = n - n_train - n_val
        splits['train'].extend(pairs[:n_train])
        splits['val'].extend(pairs[n_train:n_train + n_val])
        splits['test'].extend(pairs[n_train + n_val:n_train + n_val + n_test])
    rng.shuffle(splits['train'])
    rng.shuffle(splits['val'])
    rng.shuffle(splits['test'])
    return splits


def copy_pairs(splits: Dict[str, List[Tuple[Path, Path]]], processed_root: Path) -> Dict[str, List[Tuple[Path, Path]]]:
    out: Dict[str, List[Tuple[Path, Path]]] = {k: [] for k in splits}
    for split, pairs in splits.items():
        img_dir = processed_root / split / 'images'
        lbl_dir = processed_root / split / 'labels'
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img, lbl in pairs:
            dst_img = img_dir / img.name
            dst_lbl = lbl_dir / lbl.name
            dst_img.write_bytes(img.read_bytes())
            dst_lbl.write_bytes(lbl.read_bytes())
            out[split].append((dst_img, dst_lbl))
    return out


def sanity_check(processed_root: Path) -> None:
    for split in ['train', 'val', 'test']:
        img_dir = processed_root / split / 'images'
        lbl_dir = processed_root / split / 'labels'
        imgs = sorted(img_dir.glob('*.npy'))
        labels = sorted(lbl_dir.glob('*.npy'))
        assert len(imgs) == len(labels), f"Mismatch in {split}: {len(imgs)} imgs vs {len(labels)} labels"
        for img in imgs:
            lbl = lbl_dir / img.name.replace('env_', 'gt_')
            assert lbl.exists(), f"Missing label for {img}"


def write_manifests(splits: Dict[str, List[Tuple[Path, Path]]], processed_root: Path) -> None:
    for split, pairs in splits.items():
        manifest = processed_root / f"{split}.csv"
        with open(manifest, 'w') as f:
            f.write('image,label\n')
            for img, lbl in pairs:
                img_rel = img.relative_to(processed_root)
                lbl_rel = lbl.relative_to(processed_root)
                f.write(f"{img_rel},{lbl_rel}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description='Create full dataset')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    seed = int(cfg.get('seed', 0))
    random.seed(seed)
    np.random.seed(seed)

    raw_root = Path(cfg.get('raw_dir', 'data/raw'))
    processed_root = Path(cfg.get('processed_dir', 'data/processed'))
    raw_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)

    for archetype, count in cfg['envs_per_archetype'].items():
        if archetype not in ARCTYPES:
            continue
        arch_dir = raw_root / archetype
        arch_dir.mkdir(parents=True, exist_ok=True)
        generate_envs(archetype, int(count), arch_dir, cfg, seed)
        generate_ground_truth(arch_dir, cfg['heatmap'])

    pairs_by_arch = collect_pairs(raw_root)
    splits = split_pairs(pairs_by_arch, cfg.get('split', {}), seed)
    copied = copy_pairs(splits, processed_root)
    sanity_check(processed_root)

    if cfg.get('manifest', False):
        write_manifests(copied, processed_root)


if __name__ == '__main__':
    main()
