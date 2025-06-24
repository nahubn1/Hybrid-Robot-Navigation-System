import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import pybullet as p
import pybullet_data

SRC_PATH = Path(__file__).resolve().parents[2] / 'src'
sys.path.append(str(SRC_PATH))

from data_generation.pybullet_scene_generator import (
    create_cluttered_scene,
    create_room_scene,
    create_maze_scene,
)
from data_generation.raycast_utils import generate_occupancy_grid

ARCTYPES = ['clutter', 'room', 'maze']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate PyBullet occupancy grids')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--num-scenes', type=int, default=10)
    parser.add_argument('--archetype', type=str, choices=ARCTYPES, default='clutter')
    parser.add_argument('--mode', type=str, choices=['DIRECT', 'GUI'], default='DIRECT')
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--area-size', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--obstacle-count', type=int, default=20)
    parser.add_argument('--num-rooms', type=int, default=2)
    parser.add_argument('--maze-size', type=int, default=4)
    parser.add_argument('--passage-width', type=float, default=1.0)
    parser.add_argument('--raycast-height', type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    mode = p.GUI if args.mode.upper() == 'GUI' else p.DIRECT
    p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    asset_dir = Path(__file__).resolve().parents[2] / 'assets'
    cube_urdf = str(asset_dir / 'cube.urdf')
    wall_urdf = str(asset_dir / 'wall.urdf')

    for idx in range(args.num_scenes):
        p.resetSimulation()
        p.loadURDF('plane.urdf')
        if args.archetype == 'clutter':
            create_cluttered_scene(args.obstacle_count, args.area_size, urdf_path=cube_urdf)
        elif args.archetype == 'room':
            create_room_scene(args.num_rooms, args.area_size, urdf_path=wall_urdf)
        else:
            create_maze_scene(args.maze_size, args.passage_width, urdf_path=wall_urdf)

        grid = generate_occupancy_grid(args.area_size, args.resolution, args.raycast_height)
        grid_path = out_dir / f'scene_{idx:04d}.npy'
        np.save(grid_path, grid)
        meta = {
            'seed': args.seed,
            'archetype': args.archetype,
            'index': idx,
            'area_size': args.area_size,
            'resolution': args.resolution,
        }
        meta_path = out_dir / f'scene_{idx:04d}.json'
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    p.disconnect()


if __name__ == '__main__':
    main()
