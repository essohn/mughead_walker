from gymnasium.envs.registration import register

register(
    id="MugheadWalker-v0",
    entry_point="mughead_walker.mughead_walker:MugheadWalkerEnv",
    max_episode_steps=1600,
)
