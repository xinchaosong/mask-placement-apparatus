"""
Created on November 19, 2020

@author: Xinchao Song
"""

from .mask_placing import MaskPlacingEnv


class MaskPlacingJacoEnv(MaskPlacingEnv):
    def __init__(self):
        super(MaskPlacingJacoEnv, self).__init__(robot_type='jaco', human_control=False)


class MaskPlacingJacoHumanEnv(MaskPlacingEnv):
    def __init__(self):
        super(MaskPlacingJacoHumanEnv, self).__init__(robot_type='jaco', human_control=True)
