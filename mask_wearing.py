import gym, assistive_gym
import pybullet as p
import numpy as np
from PIL import Image
import utils
from time import sleep


def raw_sim_loop(p, env):
    while True:
        p.stepSimulation()
        joint_poses = p.calculateInverseKinematics(env.robot, 8, env.target_pos, env.target_orient)
        p.setJointMotorControlArray(bodyIndex=env.robot,
                                    jointIndices=list(range(1, 8)),
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=joint_poses,
                                    targetVelocities=np.zeros(len(joint_poses)),
                                    forces=500 * np.ones(len(joint_poses)),
                                    positionGains=0.03 * np.ones(len(joint_poses)),
                                    velocityGains=np.ones(len(joint_poses)))
        sleep(0.05)


def assisted_sim_loop(p, env):
    while True:
        env.render()
        position = env.target_pos
        orientation = env.target_orient

        # IK to get new joint positions (angles) for the robot
        target_joint_positions = p.calculateInverseKinematics(env.robot, 8, position, orientation)
        target_joint_positions = target_joint_positions[:7]

        # Get the joint positions (angles) of the robot arm
        joint_positions, joint_velocities, joint_torques = env.get_motor_joint_states(env.robot)
        joint_positions = np.array(joint_positions)[:7]

        # Set joint action to be the error between current and target joint positions
        joint_action = (target_joint_positions - joint_positions) * 10
        observation, reward, done, info = env.step(joint_action)


if __name__ == "__main__":
    env = gym.make("MaskPlacingJaco-v0")
    env.render()
    observation = env.reset()
    utils.print_joints(env, p)
    image_data = p.getCameraImage(width=600, height=800)
    img = Image.fromarray(image_data[2], 'RGB')
    img.save('my.png')
    img.show()
    print(env.target_pos)
    print(p.getEulerFromQuaternion(env.target_orient))
    numJoints = p.getNumJoints(env.robot)
    raw_sim_loop(p, env)
    # assisted_sim_loop(p, env)