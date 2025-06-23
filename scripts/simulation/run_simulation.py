import pybullet as p
import pybullet_data
import time


def main():
    # Connect to PyBullet simulator
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Load environment
    plane_id = p.loadURDF('plane.urdf')
    start_pos = [0, 0, 0.1]
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    robot_id = p.loadURDF('r2d2.urdf', start_pos, start_orientation)

    # Simple simulation loop
    for _ in range(1000):
        p.stepSimulation()
        time.sleep(1. / 240.)

    p.disconnect()


if __name__ == '__main__':
    main()
