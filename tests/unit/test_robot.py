import pybullet as p
import pybullet_data

from pathlib import Path
import sys

SRC_PATH = Path(__file__).resolve().parents[2] / 'src'
sys.path.append(str(SRC_PATH))

from simulation.robot import HolonomicRobot, LidarSensor


def test_holonomic_robot_step():
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot_id = p.loadURDF('r2d2.urdf', [0, 0, 0.1])
    robot = HolonomicRobot(robot_id)
    start_pos, _ = p.getBasePositionAndOrientation(robot_id)
    robot.set_velocity(1.0, 0.0, 0.0)
    robot.step(1.0)
    end_pos, _ = p.getBasePositionAndOrientation(robot_id)
    assert end_pos[0] != start_pos[0] or end_pos[1] != start_pos[1]
    p.disconnect()


def test_lidar_scan():
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF('plane.urdf')
    cube_id = p.loadURDF('cube_small.urdf', [1, 0, 0.5])
    robot_id = p.loadURDF('r2d2.urdf', [0, 0, 0.1])
    lidar = LidarSensor(robot_id, range=2.0, num_rays=8)
    distances = lidar.scan()
    assert len(distances) == 8
    p.disconnect()
