# Import the necessary PyTorch and Gym libraries
import torch
from torch import nn
from torchvision import transforms as T
import gym
import torch
from gym.spaces import Box
from gym.wrappers import FrameStack
import numpy as np


class SkipFrame(gym.Wrapper):

    def __init__(self, env, num_skips):
        super().__init__(env)
        self._num_skips = num_skips

    def step(self, action):
        """ Step function wrapper when frames are skipped. """
        total_reward = 0.0
        done = False
        for i in range(self._num_skips):
            observation, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return observation, total_reward, done, info


class ObserveInGrayscale(gym.ObservationWrapper):
    """ Converts an observation to grayscale. """
    def __init__(self, env):
        super().__init__(env)
        observation_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=observation_shape, dtype=np.uint8)

    def reorder_frame(self, observation):
        """ Reorders the input observation from Height, Width, Channel to Channel, Height, Width. """
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        """ Convert observation to grayscale. """
        observation=self.reorder_frame(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        observation_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=observation_shape, dtype=np.uint8)

    def observation(self, observation):
        """ Resizes and normalizes the observation. """
        transforms = T.Compose([T.Resize(self.shape), T.Normalize(0,255)])
        observation = transforms(observation).squeeze(0)
        return observation




