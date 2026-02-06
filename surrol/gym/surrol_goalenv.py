import gym
from gym import error
from surrol.gym.surrol_env import SurRoLEnv

# 兼容新旧版本的 gym/gymnasium
try:
    import gymnasium
    from gymnasium import spaces as gym_spaces
    HAS_GYMNASIUM = True
except ImportError:
    from gym import spaces as gym_spaces
    HAS_GYMNASIUM = False


class SurRoLGoalEnv(SurRoLEnv):
    """
    A gym GoalEnv wrapper for SurRoL.
    refer to: https://github.com/openai/gym/blob/master/gym/core.py
    """

    def reset(self):
        # Enforce that each GoalEnv uses a Goal-compatible observation space.
        # 兼容 gym 和 gymnasium 两种版本
        is_dict_space = isinstance(self.observation_space, gym.spaces.Dict)
        if HAS_GYMNASIUM:
            is_dict_space = is_dict_space or isinstance(self.observation_space, gymnasium.spaces.Dict)
        
        if not is_dict_space:
            raise error.Error('GoalEnv requires an observation space of type gym.spaces.Dict or gymnasium.spaces.Dict')
        for key in ['observation', 'achieved_goal', 'desired_goal']:
            if key not in self.observation_space.spaces:
                raise error.Error('GoalEnv requires the "{}" key to be part of the observation dictionary.'.format(key))
        return super().reset()
