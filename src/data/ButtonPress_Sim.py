""" 
Metaworld Button-Press Simulation
"""

import numpy as np
import torch
import torch.nn.functional as F
import metaworld
import gymnasium
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as ENVS_MW



ACTION_REPEAT = 5
CAM_CONFIG = {
    "distance": 1.25,
    "azimuth": 145,
    "elevation": -40.0,
    "lookat": np.array([-0.05, 0.75, 0.0])
}



class ButtonPressSim:
    """
    Creating a ButtonPress simulation in order to evaluate the learned models,
    behaviors and action decoders.
    
    Args:
    -----
    task_name: string
        Name of the task to perform
    num_rand_actions: int
        Number of random actions to perform for initialization
    seed: int
        Base random seed for reproducibility
    """


    def __init__(self, num_rand_actions=5, seed=10000):
        """ Module initialzer """
        self.task_name = "button_press"
        self.seed = seed
        self.num_rand_actions = num_rand_actions
        self.action_repeat = ACTION_REPEAT

        self.env = self.create_environment(seed_offset=0)
        return

    def create_environment(self, seed_offset):
        """ Creating environment """
        env = ENVS_MW["button-press-v2-goal-observable"](render_mode="rgb_array")
        env = CameraWrapper(
                env,
                seed=self.seed + seed_offset,
                camera_config=CAM_CONFIG
            )
        return env

    def _init_env(self):
        """
        Initializing an environment and executing a few random actions to randomize
        the initial position of the sawyer arm
        """
        _, _ = self.env.reset()
        for _ in range(self.num_rand_actions):
            a = self.env.action_space.sample()
            for _ in range(self.action_repeat):
                _ = self.env.step(a)
        return

    def init_seq(self, idx):
        """ Initializing a sequence"""
        self._init_env()
        img = self.env.render()
        img = self._process_observation(img)
        return img

    def update(self, action):
        """ Updating environment given action and returning the updated observation """
        if self.env is None:
            raise ValueError("Environment 'env' is not yet initialized...")
        action = self._process_action(action)
        done = False
        for _ in range(self.action_repeat):
            _, reward, _, _, info = self.env.step(action)
            if int(info['success']) == 1 and done is False:
                done = True
        img = self.env.render()
        img = self._process_observation(img)
        return img, done

    def _process_observation(self, img):
        """ Preprocessing simultated image """
        img = img.copy() / 255.
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        img = F.interpolate(img, size=(64, 64), mode="bilinear")
        return img
    
    def _process_action(self, action):
        """ Preprocessing action """
        if len(action.shape) > 1:
            assert action.shape[0] == 1
            action = action[0]
        if torch.is_tensor(action):
            action = action.cpu().numpy()
        return action



class CameraWrapper(gymnasium.Wrapper):
    """
    Module that wraps the environment so that we can set custom camera viewpoints
    and render images of our desired img_size
    """

    def __init__(self, env, seed, camera_config, img_size=224):
        """ Wrapper to set a custom camera viewpoint """
        super().__init__(env)

        self.unwrapped.model.vis.global_.offwidth = img_size
        self.unwrapped.model.vis.global_.offheight = img_size
        self.unwrapped.mujoco_renderer = MujocoRenderer(
                env.model,
                env.data,
                camera_config,
                img_size,
                img_size
            )
        self.unwrapped._freeze_rand_vec = False
        self.unwrapped.seed(seed)
        return

    def reset(self):
        """ Resetting the simulation"""
        obs, info = super().reset()
        return obs, info

    def step(self, action):
        """ Single simulation step"""
        next_obs, reward, done, truncate, info = self.env.step(action) 
        return next_obs, reward, done, truncate, info