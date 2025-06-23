import pybullet as p
import pybullet_data
import time
import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[1] / 'src'
sys.path.append(str(SRC_PATH))

from simulation.environment_generator import generate_random_2d_environment
from simulation.robot import HolonomicRobot, LidarSensor


def main():
    # Connect to PyBullet simulator
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Load environment
    plane_id = p.loadURDF('plane.urdf')
    start_pos = [0, 0, 0.1]
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    robot_id = p.loadURDF('r2d2.urdf', start_pos, start_orientation)

    robot = HolonomicRobot(robot_id)
    lidar = LidarSensor(robot_id, range=5.0, num_rays=36)

    # Generate random obstacles
    generate_random_2d_environment(num_obstacles=15, area_size=6.0)

    # Simple simulation loop
    for _ in range(1000):
        robot.set_velocity(5.0, 0.0, 0.0)
        robot.step(1.0 / 240.0)
        lidar.scan()
        time.sleep(1.0 / 240.0)

    p.disconnect()


if __name__ == '__main__':
    main()
