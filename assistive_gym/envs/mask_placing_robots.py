from .mask_placing import MaskPlacingEnv


class MaskPlacingPR2Env(MaskPlacingEnv):
    def __init__(self):
        super(MaskPlacingPR2Env, self).__init__(robot_type='pr2', human_control=False)


class MaskPlacingJacoEnv(MaskPlacingEnv):
    def __init__(self):
        super(MaskPlacingJacoEnv, self).__init__(robot_type='jaco', human_control=False)


class MaskPlacingPR2HumanEnv(MaskPlacingEnv):
    def __init__(self):
        super(MaskPlacingPR2HumanEnv, self).__init__(robot_type='pr2', human_control=True)


class MaskPlacingJacoHumanEnv(MaskPlacingEnv):
    def __init__(self):
        super(MaskPlacingJacoHumanEnv, self).__init__(robot_type='jaco', human_control=True)
