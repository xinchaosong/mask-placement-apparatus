import numpy as np
import time

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