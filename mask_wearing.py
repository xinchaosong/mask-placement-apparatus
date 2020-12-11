import gym, assistive_gym
import pybullet as p
import numpy as np
from PIL import Image
import utils
from time import sleep
import time

end_effector_id = 9

def raw_ctrl_loop(p, env, target_pos, target_orient):
    goal_pos = target_pos
    goal_orient = target_orient
    goal_orient_Euler = p.getEulerFromQuaternion(goal_orient)
    while True:
        p.stepSimulation()
        joint_poses = p.calculateInverseKinematics(env.robot, end_effector_id, goal_pos, goal_orient)
        target_joint_positions = joint_poses[:7]

        joint_positions, joint_velocities, joint_torques = env.get_motor_joint_states(env.robot)
        joint_positions = np.array(joint_positions)[:7]

        distance = np.linalg.norm(target_joint_positions - joint_positions)
        force_applied = distance

        p.setJointMotorControlArray(bodyIndex=env.robot,
                                    jointIndices=list(range(1, 8)),
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=joint_poses,
                                    targetVelocities=np.zeros(len(joint_poses)),
                                    forces=force_applied * np.ones(len(joint_poses)),
                                    positionGains=0.03 * np.ones(len(joint_poses)),
                                    velocityGains=np.ones(len(joint_poses)))
        linkPos_w, linkOrn_w, _, _, _, _ = p.getLinkState(env.robot, linkIndex=8)
        linkOrnEuler_w = p.getEulerFromQuaternion(linkOrn_w)
        # print(f'current: {linkPos_w}, {linkOrnEuler_w} | goal: {goal_pos}, {goal_orient_Euler}')
        qKey = ord('q')
        rKey = ord('r')
        keys = p.getKeyboardEvents()
        if qKey in keys and keys[qKey]&p.KEY_WAS_TRIGGERED:
            break;
        elif rKey in keys and keys[rKey]&p.KEY_WAS_TRIGGERED:
            env.reset()
        # utils.print_joints(env, p)
        print(p.getJointState(env.robot, 8)[2][0])
        sleep(0.05)
    p.disconnect()


def assisted_ctrl_loop(p, env, target_pos, target_orient):
    goal_pos = target_pos
    goal_orient = target_orient
    goal_orient_Euler = p.getEulerFromQuaternion(goal_orient)
    while True:
        env.render()

        # IK to get new joint positions (angles) for the robot
        target_joint_positions = p.calculateInverseKinematics(env.robot, end_effector_id, goal_pos, goal_orient)
        target_joint_positions = target_joint_positions[:7]

        # Get the joint positions (angles) of the robot arm
        joint_positions, joint_velocities, joint_torques = env.get_motor_joint_states(env.robot)
        joint_positions = np.array(joint_positions)[:7]

        # Set joint action to be the error between current and target joint positions
        kp = 3.0
        err = target_joint_positions - joint_positions
        u = kp*err
        observation, reward, done, info = env.step(u)
        linkPos_w, linkOrn_w, _, _, _, _ = p.getLinkState(env.robot, linkIndex=8)
        linkOrnEuler_w = p.getEulerFromQuaternion(linkOrn_w)
        # print(f'current: {linkPos_w}, {linkOrnEuler_w} | goal: {goal_pos}, {goal_orient_Euler}')
        qKey = ord('q')
        rKey = ord('r')
        keys = p.getKeyboardEvents()
        if qKey in keys and keys[qKey]&p.KEY_WAS_TRIGGERED:
            break;
        elif rKey in keys and keys[rKey]&p.KEY_WAS_TRIGGERED:
            env.reset()
            goal_pos = env.target_pos
            goal_orient = env.target_orient
        utils.print_joints(env, p)
        # print(p.getJointState(env.robot, 8)[2][0])
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

# States
FIND_POSE = 0
MASK_ON = 1
MASK_OFF = 2
DONE = 3

class PID:
    def __init__(self, kp=3.0, ki=0.0001, kd=0.0001, q_dim=1, current_time=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.current_time = current_time if current_time is not None else time.time()
        self.last_time = self.current_time
        self.q_dim = q_dim
        self.initialize()

    def initialize(self, current_time=None):
        self.PTerm = np.zeros(self.q_dim).tolist()
        self.ITerm = np.zeros(self.q_dim).tolist()
        self.DTerm = np.zeros(self.q_dim).tolist()
        self.last_error = np.zeros(self.q_dim).tolist()
        self.int_error = np.zeros(self.q_dim).tolist()
        self.windup_guard = 20.0

    def update(self, error, current_time=None):
        self.current_time = current_time if current_time is not None else time.time()
        delta_time = self.current_time - self.last_time
        delta_error = error - self.last_error

        self.PTerm = self.kp * error
        self.ITerm += error * delta_time
        for i in range(self.q_dim):
            if (self.ITerm[i] < -self.windup_guard):
                self.ITerm[i] = -self.windup_guard
            elif (self.ITerm[i] > self.windup_guard):
                self.ITerm[i] = self.windup_guard
        
        self.DTerm = 0.0
        if delta_time > 0:
            self.DTerm = delta_error / delta_time

        self.last_time = self.current_time
        self.last_error = error
        return self.PTerm + (self.ki * self.ITerm) + (self.kd * self.DTerm)


def run():
    env = gym.make("MaskPlacingJaco-v0")
    env.render()
    observation = env.reset()
    utils.print_joints(env, p)
    p.addUserDebugLine(lineFromXYZ=[0.0, 0.0, 0.0], lineToXYZ=[0.0, 0.0, 2.0], lineColorRGB=[255,0,0])
    p.addUserDebugLine(lineFromXYZ=[0.25, -0.5, 0.0], lineToXYZ=[0.25, -0.5, 0.75], lineColorRGB=[0,0,255])

    qKey = ord('q')
    rKey = ord('r')
    fKey = ord('f')
    bKey = ord('b')

    # FSM state variable
    currState = FIND_POSE

    target_pos = env.target_pos
    target_orient = env.target_orient

    startPos, startOrient, _, _, _, _ = p.getLinkState(env.robot, linkIndex=end_effector_id)

    pid_ctrl = PID(kp=3.0, ki=0.0, kd=0.0, q_dim=7)

    # Main loop
    while True:
        if currState == FIND_POSE:
            currState = MASK_ON
        elif currState == MASK_ON:
            env.render()
            # IK to get new joint positions (angles) for the robot
            target_joint_positions = p.calculateInverseKinematics(env.robot, end_effector_id, target_pos, target_orient)
            target_joint_positions = target_joint_positions[:7]
            # Get the joint positions (angles) of the robot arm
            joint_positions, joint_velocities, joint_torques = env.get_motor_joint_states(env.robot)
            joint_positions = np.array(joint_positions)[:7]
            # Set joint action to be the error between current and target joint positions

            err = target_joint_positions - joint_positions
            curr_time = time.time()
            u = pid_ctrl.update(err)
            observation, reward, done, info = env.step(u)
            # p.stepSimulation()
            # joint_poses = p.calculateInverseKinematics(env.robot, end_effector_id, target_pos, target_orient)
            # target_joint_positions = joint_poses[:7]

            # joint_positions, joint_velocities, joint_torques = env.get_motor_joint_states(env.robot)
            # joint_positions = np.array(joint_positions)[:7]

            # distance = np.linalg.norm(target_joint_positions - joint_positions)
            # force_applied = distance*0.25

            # p.setJointMotorControlArray(bodyIndex=env.robot,
            #                             jointIndices=list(range(1, 8)),
            #                             controlMode=p.POSITION_CONTROL,
            #                             targetPositions=joint_poses,
            #                             targetVelocities=np.zeros(len(joint_poses)),
            #                             forces=force_applied * np.ones(len(joint_poses)),
            #                             positionGains=0.03 * np.ones(len(joint_poses)),
            #                             velocityGains=np.ones(len(joint_poses)))
            # sleep(0.05)

        elif currState == MASK_OFF:
            # u = actions_taken.pop()
            # observation, reward, done, info = env.step(u)
            # if len(actions_taken) == 0:
            #     currState = DONE
            p.stepSimulation()
            joint_poses = p.calculateInverseKinematics(env.robot, end_effector_id, target_pos, target_orient)
            target_joint_positions = joint_poses[:7]

            joint_positions, joint_velocities, joint_torques = env.get_motor_joint_states(env.robot)
            joint_positions = np.array(joint_positions)[:7]

            distance = np.linalg.norm(target_joint_positions - joint_positions)
            # force_applied = distance*0.25
            force_applied = 1.0

            p.setJointMotorControlArray(bodyIndex=env.robot,
                                        jointIndices=list(range(1, 8)),
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=joint_poses,
                                        targetVelocities=np.zeros(len(joint_poses)),
                                        forces=force_applied * np.ones(len(joint_poses)),
                                        positionGains=0.03 * np.ones(len(joint_poses)),
                                        velocityGains=np.ones(len(joint_poses)))
            sleep(0.05)
        else:
            pass
        
        # State transition logic
        keys = p.getKeyboardEvents()
        if qKey in keys and keys[qKey]&p.KEY_WAS_TRIGGERED: # Quit
            break;
        elif fKey in keys and keys[fKey]&p.KEY_WAS_TRIGGERED: # Forward
            currState = MASK_ON
            target_pos = env.target_pos
            target_orient = env.target_orient
        elif bKey in keys and keys[bKey]&p.KEY_WAS_TRIGGERED: # Backward
            # currState = MASK_OFF
            target_pos = startPos
            target_orient = startOrient

        elif rKey in keys and keys[rKey]&p.KEY_WAS_TRIGGERED: # Reset Env
            env.reset()
            goal_pos = env.target_pos
            goal_orient = env.target_orient
            currState = FIND_POSE
    
    p.disconnect()

if __name__ == "__main__":
    run()
