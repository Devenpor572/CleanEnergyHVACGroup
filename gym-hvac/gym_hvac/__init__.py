from gym.envs.registration import register

register(
    id='HVAC-v0',
    entry_point='gym_hvac.envs:HVACEnv',
)
