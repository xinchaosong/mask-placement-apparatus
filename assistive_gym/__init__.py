from gym.envs.registration import register

register(
    id='MaskPlacingJaco-v0',
    entry_point='assistive_gym.envs:MaskPlacingJacoEnv',
    max_episode_steps=200,
)

register(
    id='MaskPlacingJacoHuman-v0',
    entry_point='assistive_gym.envs:MaskPlacingJacoHumanEnv',
    max_episode_steps=200,
)
