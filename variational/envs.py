import os



import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from baselines import bench
from AI_scientist.variational.vec_env.atari_wrappers import make_atari, wrap_deepmind

import gym
from gym.spaces.box import Box

try:
    import pybullet_envs
except ImportError:
    pass


def make_env(env_id, seed, rank, log_dir, clip_rewards = True, env_settings = {}):
    def _thunk():
        if "Custom" in env_id:
            env = gym.make(env_id, env_settings = env_settings)
        else:
            env = gym.make(env_id)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
        env.seed(seed + rank)
        env = bench.Monitor(env, os.path.join(log_dir, str(rank)))
        if is_atari:
            env = wrap_deepmind(env, clip_rewards = clip_rewards, env_settings = env_settings)
            env = WrapPyTorch(env, env_settings = env_settings)
        return env

    return _thunk


class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None, env_settings = {}):
        super(WrapPyTorch, self).__init__(env)
        width = 84 if "width" not in env_settings else env_settings["width"]
        height = 84 if "height" not in env_settings else env_settings["height"]
        self.observation_space = Box(0.0, 1.0, [1, height, width])

    def _observation(self, observation):
        return observation.transpose(2, 0, 1)
