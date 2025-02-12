""" 
BlockPush Simulation
"""

import torch
from PIL import Image
from torchvision import transforms

import logging
import gym
import multi_object_fetch


ACTION_REPEAT = 2  # to match expert-demos setup


class BlockPushSim:
    """
    Creating a BlockPush simulation in order to evaluate the learned models,
    behaviors and action decoders.
    
    Args:
    -----
    seed: int
        Base random seed for reproducibility
    """
    
    def __init__(self, task_name, seed=10000):
        """ Module initialzer """
        gym.logger.set_level(logging.ERROR)
        self.task_name = task_name
        self.seed = seed
        self.action_repeat = ACTION_REPEAT
        self.env = None                
        return

    def _init_env(self, seed_offset):
        """ Initializing an environment """
        env = gym.make(self.task_name)
        env.seed(self.seed + seed_offset)
        _ = env.reset()
        return env

    def init_seq(self, idx):
        """ Initializing a sequence"""
        self.env = self._init_env(seed_offset=idx)
        img = self.env.render()
        img = self._process_observation(img)
        return img

    def update(self, action):
        """ Updating environment given action and returning the updated observation """
        if self.env is None:
            raise ValueError("Environment 'env' is not yet initialized...")
        action = self._process_action(action)

        for _ in range(self.action_repeat):
            _, _, _, _ = self.env.step(action)
        done = int(self.env.success()) == 1

        img = self.env.render()
        img = self._process_observation(img)
        return img, done

    def _process_observation(self, img):
        """ Preprocessing simultated image """
        img = Image.fromarray(img)
        img = img.resize((64, 64))
        img = transforms.ToTensor()(img)[:3].unsqueeze(dim=0)
        return img
    
    def _process_action(self, action):
        """ Preprocessing action """
        if len(action.shape) > 1:
            assert action.shape[0] == 1
            action = action[0]
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        return action
