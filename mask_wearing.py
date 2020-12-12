import gym, assistive_gym
import pybullet as p
import numpy as np
from PIL import Image
import utils
from time import sleep
import time
import cv2
import dlib
import math

end_effector_id = 9
# States
FIND_POSE = 0
MASK_ON = 1
MASK_OFF = 2
DONE = 3

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

def getCameraFrame(p):
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
    return image_data[2]

def StartVideo():
    video_capture = cv2.VideoCapture(0)
    cv2.namedWindow("Window")
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    count = 0
    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        ## Facial detection
        faces = detector(gray)
        #print(len(faces)) # =1
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1) #1 pixel thickness
            
            landmarks = predictor(gray, face)
            
            '''
            ### Find the ears upper and lower region
            #for n in range(0, 68):
            for n in [0, 2, 14, 16]:      ### Ears = [0,2,14,16]
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x,y), 3, (255, 0, 0), -1) #radius=3, fill the circle = -1
            '''   
        
              
        ### Drawing a line from point 28 to 8.
        x28 = landmarks.part(28).x
        y28 = landmarks.part(28).y
        x8 = landmarks.part(8).x
        y8 = landmarks.part(8).y
        
        
        ### Calculate angle between the 2 points
        angle = math.atan2((y8-y28),(x8-x28))*180/3.141592653;
        angle = angle + math.ceil(-angle / 360 ) * 360
        
        allx = []
        ally = []
        if angle > 85 and angle < 95:
            
            x = landmarks.part(66).x
            y = landmarks.part(66).y
            
            allx.append(x)
            ally.append(y)
            
            cv2.putText(frame, "Perfect, stay still!, mouth location = ({},{})".format(x, y),
                    (0,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            

            cv2.circle(frame, (x,y), 4, (255, 0, 0), -1) #radius=3, fill the circle = -1 
            #return x, y
        
        else:
            
            ### Facial Center points forehead, lower lip center and chin. Points 28, 57 and 8.
            for n in [28, 57, 8]:
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x,y), 4, (255, 0, 0), -1) #radius=3, fill the circle = -1
            
            
            cv2.line(frame, (x28, y28), (x8,y8), (0,0,0), 2) 


            ### Tell the user to adjust the face orientation
            #cv2.putText(img,'Hello World!', bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
            cv2.putText(frame,'Adjust face to 90 Deg, current angle = {}'.format(angle),
                        (0,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                
        
        cv2.imshow("Window", frame)

        #This breaks on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    return math.floor(np.mean(allx)), math.floor(np.mean(ally))

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

def assisted_ctrl_loop(p, env, target_pos, target_orient, pid_ctrl):
    env.render()
    # IK to get new joint positions (angles) for the robot
    target_joint_positions = p.calculateInverseKinematics(env.robot, end_effector_id, target_pos, target_orient)
    target_joint_positions = target_joint_positions[:7]
    # Get the joint positions (angles) of the robot arm
    joint_positions, joint_velocities, joint_torques = env.get_motor_joint_states(env.robot)
    joint_positions = np.array(joint_positions)[:7]
    # Set joint action to be the error between current and target joint positions
    err = target_joint_positions - joint_positions
    u = pid_ctrl.update(err)
    observation, reward, done, info = env.step(u)


def raw_ctrl_loop(p, env, target_pos, target_orient, reduce_force=True):
    p.stepSimulation()
    joint_poses = p.calculateInverseKinematics(env.robot, end_effector_id, target_pos, target_orient)
    target_joint_positions = joint_poses[:7]

    joint_positions, joint_velocities, joint_torques = env.get_motor_joint_states(env.robot)
    joint_positions = np.array(joint_positions)[:7]

    if reduce_force:
        distance = np.linalg.norm(target_joint_positions - joint_positions)
        force_applied = distance*0.25
    else:
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

def run():
    env = gym.make("MaskPlacingJaco-v0")

    # Set head orientation
    env.set_head_orient(0.0, 0.0, 30.0)
    
    # Set robot base position (relative to wheelchair)
    env.set_robot_base([-0.35, -0.3, 0.3]) # x, y, z in meters

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
            frame = getCameraFrame(p)
            img = Image.fromarray(frame, 'RGBA')
            img.save('head.png')
            # x, y = StartVideo()
            # print(x, y)
            currState = MASK_ON
        elif currState == MASK_ON:
            assisted_ctrl_loop(p, env, target_pos, target_orient, pid_ctrl)
            # raw_ctrl_loop(p, env, target_pos, target_orient, reduce_force=True)

        elif currState == MASK_OFF:
            assisted_ctrl_loop(p, env, startPos, startOrient, pid_ctrl)
            # raw_ctrl_loop(p, env, startPos, startOrient, reduce_force=False)
        else:
            pass
        
        # State transition logic
        keys = p.getKeyboardEvents()
        if qKey in keys and keys[qKey]&p.KEY_WAS_TRIGGERED: # Quit
            break;
        elif fKey in keys and keys[fKey]&p.KEY_WAS_TRIGGERED: # Forward
            currState = MASK_ON
        elif bKey in keys and keys[bKey]&p.KEY_WAS_TRIGGERED: # Backward
            currState = MASK_OFF
        elif rKey in keys and keys[rKey]&p.KEY_WAS_TRIGGERED: # Reset Env
            env.reset()
            currState = FIND_POSE
    
    p.disconnect()

if __name__ == "__main__":
    run()
