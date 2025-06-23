import random
import pybullet as p


def generate_random_2d_environment(num_obstacles=10, area_size=5.0, obstacle_size_range=(0.2, 0.5)):
    """Generate a 2D environment with random box obstacles.

    Parameters
    ----------
    num_obstacles : int
        Number of obstacles to generate.
    area_size : float
        Size of the square area in which obstacles are placed.
    obstacle_size_range : tuple(float, float)
        Range of box sizes (edge length).

    Returns
    -------
    list of int
        PyBullet IDs of the created obstacles.
    """
    obstacle_ids = []
    half_area = area_size / 2.0
    for _ in range(num_obstacles):
        size = random.uniform(*obstacle_size_range)
        half_size = size / 2.0
        x = random.uniform(-half_area, half_area)
        y = random.uniform(-half_area, half_area)
        z = half_size
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half_size]*3)
        vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[half_size]*3,
                                        rgbaColor=[0.6, 0.6, 0.6, 1.0])
        body_id = p.createMultiBody(baseMass=0,
                                    baseCollisionShapeIndex=col_shape,
                                    baseVisualShapeIndex=vis_shape,
                                    basePosition=[x, y, z])
        obstacle_ids.append(body_id)
    return obstacle_ids
