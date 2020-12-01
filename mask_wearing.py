import gym, assistive_gym
import pybullet as p
import numpy as np
from PIL import Image
import utils
from time import sleep


def raw_ctrl_loop(p, env, target_pos, target_orient):
    goal_pos = target_pos
    goal_orient = target_orient
    goal_orient_Euler = p.getEulerFromQuaternion(goal_orient)
    while True:
        p.stepSimulation()
        joint_poses = p.calculateInverseKinematics(env.robot, 8, goal_pos, goal_orient)
        p.setJointMotorControlArray(bodyIndex=env.robot,
                                    jointIndices=list(range(1, 8)),
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=joint_poses,
                                    targetVelocities=np.zeros(len(joint_poses)),
                                    forces=500 * np.ones(len(joint_poses)),
                                    positionGains=0.03 * np.ones(len(joint_poses)),
                                    velocityGains=np.ones(len(joint_poses)))
        linkPos_w, linkOrn_w, _, _, _, _ = p.getLinkState(env.robot, linkIndex=8)
        linkOrnEuler_w = p.getEulerFromQuaternion(linkOrn_w)
        print(f'current: {linkPos_w}, {linkOrnEuler_w} | goal: {goal_pos}, {goal_orient_Euler}')
        qKey = ord('q')
        keys = p.getKeyboardEvents()
        if qKey in keys and keys[qKey]&p.KEY_WAS_TRIGGERED:
            break;
        sleep(0.05)
    p.disconnect()


def assisted_ctrl_loop(p, env, target_pos, target_orient):
    goal_pos = target_pos
    goal_orient = target_orient
    goal_orient_Euler = p.getEulerFromQuaternion(goal_orient)
    while True:
        env.render()

        # IK to get new joint positions (angles) for the robot
        target_joint_positions = p.calculateInverseKinematics(env.robot, 8, goal_pos, goal_orient)
        target_joint_positions = target_joint_positions[:7]

        # Get the joint positions (angles) of the robot arm
        joint_positions, joint_velocities, joint_torques = env.get_motor_joint_states(env.robot)
        joint_positions = np.array(joint_positions)[:7]

        # Set joint action to be the error between current and target joint positions
        kp = 10.0
        err = target_joint_positions - joint_positions
        u = kp*err
        observation, reward, done, info = env.step(u)
        linkPos_w, linkOrn_w, _, _, _, _ = p.getLinkState(env.robot, linkIndex=8)
        linkOrnEuler_w = p.getEulerFromQuaternion(linkOrn_w)
        print(f'current: {linkPos_w}, {linkOrnEuler_w} | goal: {goal_pos}, {goal_orient_Euler}')
        qKey = ord('q')
        keys = p.getKeyboardEvents()
        if qKey in keys and keys[qKey]&p.KEY_WAS_TRIGGERED:
            break;
    p.disconnect()

def getHeadPoseFromCamera(p):
    # Camera setup
    camera_pos = [0.25, -0.5, 0.75]
    target_pos = [0.0, 0.0, 1.25]
    up_vector = [0.0, 0.0, 1.0]
    viewMatrix = p.computeViewMatrix(cameraEyePosition=camera_pos,
                                     cameraTargetPosition=target_pos,
                                     cameraUpVector=up_vector)
    width = 640
    height = 480
    fov = 60
    aspect = width / height
    near = 0.02
    far = 1
    projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    image_data = p.getCameraImage(height=height,
                                width=width,
                                viewMatrix=viewMatrix,
                                projectionMatrix=projectionMatrix,
                                shadow=True,
                                renderer=p.ER_BULLET_HARDWARE_OPENGL)
    img = Image.fromarray(image_data[2], 'RGBA')
    img.save('my.png')
    # TODO return head pose using face recognition algorithm                         
    target_orient = p.getEulerFromQuaternion(env.target_orient)
    return env.target_pos, target_orient, True


if __name__ == "__main__":
    env = gym.make("MaskPlacingJaco-v0")
    env.render()
    observation = env.reset()
    utils.print_joints(env, p)
    p.addUserDebugLine(lineFromXYZ=[0.0, 0.0, 0.0], lineToXYZ=[0.0, 0.0, 2.0], lineColorRGB=[255,0,0])
    p.addUserDebugLine(lineFromXYZ=[0.25, -0.5, 0.0], lineToXYZ=[0.25, -0.5, 0.75], lineColorRGB=[0,0,255])

    ready = False

    while not ready:
        target_pos, target_orient, ready = getHeadPoseFromCamera(p)

    print(target_pos)
    print(target_orient)
    print(target_orient)
    numJoints = p.getNumJoints(env.robot)
    # raw_ctrl_loop(p, env, target_pos, p.getQuaternionFromEuler(target_orient))
    assisted_ctrl_loop(p, env, target_pos, p.getQuaternionFromEuler(target_orient))