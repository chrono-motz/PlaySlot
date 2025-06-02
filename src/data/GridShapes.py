"""
Grid Shapes dataset.

One or more shapes are placed in the image, and they can move in one of
four directions (top, down, left, right).

They have more probability of moving in the same direction as they are moving
but can randomly change directions with certain probability.
"""

import os
import random
import numpy as np 
import cv2
import torch
from torch.utils.data import Dataset
from webcolors import name_to_rgb



class BaseGridShapes(Dataset):
    """
    Base Grid Shapes dataset that generates data on-the-fly.
    This module is used to generate the dataset, which can be then
    loaded by the module below
    
    Args:
    -----
    num_frames: int
        Number of frames to render in the sequence
    num_balls: int
        Number of shapes in the sequence
    img_size: int
        Size of the image to generate. We assume square images
    change_prob: float
        Probability of an object changing direction
    shape_size: int
        Size of the shapes
    use_bkgr: bool
        If True, a random color is addded as background
    """

    MOVING_SPECS = {
        "speed_min": 3,
        "speed_max": 3,
        "acc_min": 0,
        "acc_max": 0
    }
    COLORS = {
        0: "red",
        1: "cyan",
        2: "green",
        3: "blue",
        4: "magenta",
        5: "yellow",
        6: "orange",
        7: "purple",
        8: "white",
        9: "brown"
    }
    SHAPE = {
        0: "ball",
        1: "triangle",
        2: "square"
    }
    PROPERTIES = {
        "shape": {
            "type": "categorical",
            "num_values": len(SHAPE.keys())
        },
        "color": {
            "type": "categorical",
            "num_values": len(COLORS.keys())
        },
        "positions": {
            "type": "continuous",
            "num_values": 2
        },
        "velocities": {
            "type": "temporal",
            "num_values": 2
        }
    }

    def __init__(self, num_frames=30, num_balls=3, img_size=64, change_prob=0.25,
                 shape_size=21, use_bkgr=False):
        """ Initializer of the GridShapes dataset """
        self.num_frames = num_frames
        self.num_balls = num_balls
        self.img_size = img_size
        self.use_bkgr = use_bkgr
        self.shape_size = shape_size
        self.change_prob = change_prob

        # loading moving parameters
        self.get_init_pos = lambda img_size, shape_size: np.random.randint(0, img_size - shape_size)
        self.get_direction = lambda: np.random.choice(["horizontal", "vertical"])
        self.get_mult = lambda: np.random.choice([-1, 1])
        self.get_color = lambda: np.random.choice(list(self.COLORS.values()))
        self.get_shape = lambda: np.random.choice(list(self.SHAPE.values()))
        return

    def get_speed(self):
        """ Sampling moving speed of the objects """
        speed_max = self.MOVING_SPECS["speed_max"]
        speed_min = self.MOVING_SPECS["speed_min"]
        speed = np.random.randint(speed_min, speed_max+1)
        return speed


    def __len__(self):
        """ """
        raise NotImplementedError("Base class does not implement __len__ functionality")


    def __getitem__(self, idx):
        """
        Sampling sequence
        """
        raise NotImplementedError("Base class does not implement __getitem__ functionality")
    
    
    def init_seq(self):
        """ Initializing sequence """
        shapes, next_poses, speeds, obj_metas = [], [], [], []
        for i in range(self.num_balls):
            shape, pos, speed, meta = self._sample_shape()
            meta["depth"] = self.num_balls - i - 1
            shapes.append(shape)
            next_poses.append(pos)
            speeds.append(speed)
            obj_metas.append(meta)
        return shapes, next_poses, speeds, obj_metas


    def init_canvas(self, obj_metas):
        """ 
        Initializing image canvs for the sequence.
        Using a background color that must differ from object colors
        """
        if self.use_bkgr:
            color_names = [m["color"] for m in obj_metas]
            bkgr_color_name = random.choice(
                    [c for c in self.COLORS.values() if c not in color_names] + ["black"]
                )
        else:
            bkgr_color_name = "black"
        bkgr_color = torch.tensor(name_to_rgb(bkgr_color_name)).float() / 255

        frames = np.ones((self.num_frames, 3, self.img_size, self.img_size))
        frames = frames* bkgr_color.view(3, 1, 1).numpy()
        return frames
    
    
    def generate_new_seq(self):
        """ Generating a new sequence of the datasets """
        # initial conditions
        shapes, next_poses, speeds, obj_metas = self.init_seq()
        frame = self.init_canvas(obj_metas)
        
        speeds_per_frame = torch.empty(self.num_frames, self.num_balls, 2)
        pos_per_frame = torch.empty(self.num_frames, self.num_balls, 2)
        
        # generating sequence by moving the shapes given velocity
        for i, frame in enumerate(frames):
            for j, (digit, cur_pos, speed) in enumerate(zip(shapes, next_poses, speeds)):
                # moving shape
                speed = self._check_speed_reroll(speed)
                digit_size = digit.shape[-2]
                speed, cur_pos = self._move_shape(
                        speed=speed,
                        cur_pos=cur_pos,
                        img_size=self.img_size,
                        digit_size=digit_size
                    )
                speeds[j] = speed
                next_poses[j] = cur_pos

                # inserting shape into canvas
                idx = digit.sum(dim=0) > 0
                frame[
                        :,
                        cur_pos[0]:cur_pos[0]+digit_size,
                        cur_pos[1]:cur_pos[1]+digit_size
                    ][:, idx] = digit[:, idx]
                speeds_per_frame[i, j, 0] = speed[0]
                speeds_per_frame[i, j, 1] =  speed[1]
                pos_per_frame[i, j, 0] = cur_pos[0]
                pos_per_frame[i, j, 1] = cur_pos[1]
            frames[i] = np.clip(frame, 0, 1)
        frames = torch.Tensor(frames)
        
        motion_metas = {
            "positions": pos_per_frame,
            "speeds": speeds_per_frame,
        }
        return frames, obj_metas, motion_metas


    def _check_speed_reroll(self, speed):
        """ Changing moving direction given a certain probability """       
        # randomly changing direction
        sample = np.random.rand()
        if sample < self.change_prob:
            cur_speed = speed[0] if speed[0] != 0 else speed[1]
            speed = self._sample_speed(speed=cur_speed)            
        return speed


    def _sample_shape(self):
        """ Sampling shape, original position and speed """
        shape_name = self.get_shape()
        color_name = self.get_color()
        shape = self._make_shape(shape_name=shape_name, color_name=color_name)

        # obtaining position in original frame and inital velocity
        x_coord = self.get_init_pos(self.img_size, self.shape_size)
        y_coord = self.get_init_pos(self.img_size, self.shape_size)
        cur_pos = np.array([y_coord, x_coord])
        speed = self._sample_speed()
        meta = {
            "color": color_name,
            "shape": shape_name,
            "init_speed": speed,
            "init_pos": torch.from_numpy(cur_pos)
        }
        return shape, cur_pos, speed, meta


    def _sample_speed(self, speed=None, use_dir=None,
                      random_mult=True, random_speed=False):
        """ Sample a new speed """
        # sampling speed direction
        if use_dir is None:
            direction = self.get_direction()
        else:
            assert use_dir in ["horizontal", "vertical"]
            direction = use_dir                
                
        # multiplier for changing direction
        if random_mult:    
            mult = self.get_mult()
        else:
            mult = 1 if any([s > 0 for s in speed]) else -1
        
        # updating speed magnitude
        if random_speed is False and speed is not None:
            base_speed = speed
        else:
            base_speed = self.get_speed()
        speed = base_speed * mult
        
        if direction == "horizontal":
            speed = np.array([0, speed])
        else:
            speed = np.array([speed, 0])
        return speed


    def _move_shape(self, speed, cur_pos, img_size, digit_size):
        """
        Performing a shape movement.
        Also producing bounce and making appropriate changes
        """
        next_pos = cur_pos + speed
        if next_pos[0] < 0:
            next_pos[0] = 0
            speed[0] = -1 * speed[0]           
        elif next_pos[0] > img_size - digit_size:
            next_pos[0] = img_size - digit_size - 1
            speed[0] = -1 * speed[0]
                        
        if next_pos[1] < 0:
            next_pos[1] = 0
            speed[1] = -1 * speed[1]
        elif next_pos[1] > img_size - digit_size:
            next_pos[1] = img_size - digit_size - 1
            speed[1] = -1 * speed[1]          
        return speed, next_pos


    def _make_shape(self, shape_name, color_name, size=None):
        """  Creating a shape """
        size = size if size is not None else self.shape_size
        
        aux = torch
        aux = np.zeros((size, size))
        size_half = size // 2
        if shape_name == "ball":
            shape = cv2.circle(aux, (size_half, size_half), int(size_half), 1, -1)
        elif shape_name == "square":
            shape = cv2.rectangle(aux, (0,0), (size, size), 1, -1)
        elif shape_name == "triangle":
            coords = np.array([[size_half, 0], [0,size], [size,size]])
            coords = coords.reshape((-1, 1, 2))
            shape = cv2.fillPoly(aux, [coords], 255, 1)
        else:
            raise ValueError(f"Unkwnown {shape_name = }...")
        shape = torch.Tensor(shape).unsqueeze(0).repeat(3, 1, 1)
        color = torch.tensor(name_to_rgb(color_name)).float() / 255
        shape = (shape * color.view(-1, 1, 1))
        return shape



class GridShapes(BaseGridShapes):
    """ 
    Class to generate the GridShapes dataset, featuring various shapes of different
    colors and sizes.
    """
    
    
    def __init__(self, split, num_frames=30, num_balls=2, **kwargs):
        """ Initializer of the GridShapes dataset """
        self.split = split
        self.max_speed = 2
        super().__init__(
            num_frames=num_frames,
                num_balls=num_balls,
                img_size=64,
                change_prob=0,
                shape_size=15,
                use_bkgr=True
            )
        return
    
    def get_speed(self):
        """ Sampling moving speed of the objects """
        speed = int(2 * (self.img_size / 64))
        return speed

    def _sample_speed(self):
        """ Sampling moving speed of the objects """
        speed = int(2 * (self.img_size / 64))
        speed_v = [speed, 0] if np.random.rand() > 0.5 else [0, speed]
        return speed_v

    def __len__(self):
        """ """
        if self.split == "train":
            N = 10000
        else:
            N = 500
        return N

    def get_new_item(self, idx):
        """ Creating new sequence """
        frames, obj_metas, motion_metas = self.generate_new_seq() 
        # converting object properties labels to IDs
        colors, shapes = [], []
        for i, cur_obj_meta in enumerate(obj_metas):
            cur_colors = list(self.COLORS.values()).index(cur_obj_meta["color"]) 
            cur_shapes = list(self.SHAPE.values()).index(cur_obj_meta["shape"])
            colors.append(torch.tensor(cur_colors))
            shapes.append(torch.tensor(cur_shapes))
           
        meta = {
            "color": torch.stack(colors),
            "shape": torch.stack(shapes),
            "actions": motion_metas["actions"],
            "positions": motion_metas["positions"] / 43,  # normalized to [0, 1]
            "velocities": motion_metas["speeds"] / self.MOVING_SPECS["speed_max"] # norm to [-1, 1]
        }
        return frames, frames, meta

    def __getitem__(self, idx):
        """ """
        if self.split == "test":
            path = f"./datasets/GridShapes/{self.num_balls}Objs"
            frames = torch.load(os.path.join(path, f"{idx:04}.pt"))
            meta = {}
        else:
            frames, frames, meta = self.get_new_item(idx)
        return frames, frames, meta


    def generate_new_seq(self):
        """
        Generating a new sequence of the datasets
        """
        # initial conditions
        shapes, next_poses, speeds, obj_metas = self.init_seq()
        frames = self.init_canvas(obj_metas)
        
        speeds_per_frame = torch.empty(self.num_frames, self.num_balls, 2)
        pos_per_frame = torch.empty(self.num_frames, self.num_balls, 2)
        actions = torch.empty(self.num_frames, self.num_balls)

        # generating sequence by moving the shapes given velocity
        for i, frame in enumerate(frames):
            # adding shapes
            for j, (digit, cur_pos, speed) in enumerate(zip(shapes, next_poses, speeds)):
                digit_size = digit.shape[-2]
                speed, cur_pos, action = self._move_shape(
                        speed=speed,
                        cur_pos=cur_pos,
                        img_size=self.img_size,
                        digit_size=digit_size,
                    )
                speeds[j] = speed
                next_poses[j] = cur_pos

                idx = digit.sum(dim=0) > 0
                frame[:, cur_pos[0]:cur_pos[0]+digit_size, cur_pos[1]:cur_pos[1]+digit_size][:, idx] = digit[:, idx]

                speeds_per_frame[i, j, 0], speeds_per_frame[i, j, 1] = speed[0], speed[1]
                pos_per_frame[i, j, 0], pos_per_frame[i, j, 1] = cur_pos[0], cur_pos[1]
                actions[i, j] = action
            frames[i] = np.clip(frame, 0, 1)
        frames = torch.Tensor(frames)

        # actions =  self._speed_to_action(speeds_per_frame)
        motion_metas = {
            "positions": pos_per_frame,
            "speeds": speeds_per_frame,
            "actions": actions
        }
        return frames, obj_metas, motion_metas


    def _speed_to_action(self, speeds):
        """ Converting (x, y) speeds into normalized action embeddings """
        max_speed = self.max_speed
        actions = speeds / max_speed
        return actions

    def expert_speed_update(self, speed, cur_pos, digit_size, target_pos):
        """ Moving object towards target """
        # offset for adjusting center
        target_pos_aux = [
                target_pos[0] - round(digit_size/2) + self.target_size // 2,
                target_pos[1] - round(digit_size/2) + self.target_size // 2
            ]
        dist_y = (target_pos_aux[0] - cur_pos[0])
        dist_x = (target_pos_aux[1] - cur_pos[1])
        
        # we move towards the target along the dimension where it is the furthest
        base_speed = self.get_speed()
        if np.abs(dist_y) != 0:
            speed_y = base_speed if dist_y >=0 else -base_speed
            speed_x = 0
        else:
            speed_x = base_speed if dist_x >=0 else -base_speed
            speed_y = 0
        
        move_y = speed_y if np.abs(speed_y) < np.abs(dist_y) else dist_y
        move_x = speed_x if np.abs(speed_x) < np.abs(dist_x) else dist_x
        updated_speed = [move_y, move_x]
        
        action = self._get_action_from_speed(updated_speed)
        
        return updated_speed, action

    def random_speed_update(self, speed, change_prob=0.25):
        """ Random speed update by change of direction"""
        # grid move: can reverse or change
        sample_change = np.random.rand()
        sample_new = np.random.rand()
        if sample_change < change_prob:
            # computing speed magnitude
            speed_mag = speed[0] if speed[0] != 0 else speed[1]
            if speed_mag == 0:
                speed_mag = self.get_speed()
            # selecting moving direction
            if sample_new < 0.20:
                speed = [speed_mag, 0]
            elif sample_new >= 0.2 and sample_new < 0.4:
                speed = [-1 * speed_mag, 0]
            elif sample_new >= 0.4 and sample_new < 0.6:
                speed = [0, speed_mag]
            elif sample_new >= 0.6 and sample_new < 0.8:
                speed = [0, -1 * speed_mag]
            elif sample_new >= 0.8:
                speed = [0, 0]
        action = self._get_action_from_speed(speed)
        return speed, action

    def _get_action_from_speed(self, speed):
        """ Computing a discrete action given the moving speed"""
        speed_y, speed_x = speed[0], speed[1]
        if speed_x > 0 and speed_y == 0:
            action = 0
        elif speed_x < 0 and speed_y == 0:
            action = 1
        elif speed_y > 0 and speed_x == 0:
            action = 2
        elif speed_y < 0 and speed_x == 0:
            action = 3
        elif speed_x == 0 and speed_y == 0:
            action = 4
        else:
            raise ValueError(f"Weird {speed = }")
        return action

  
    def _move_shape(self, speed, cur_pos, img_size, digit_size, update_speed=True):
        """ Moving digit towards its goal location """
        if update_speed:
            updated_speed, action = self.random_speed_update(speed, change_prob=0.25)
        else:        
            updated_speed = speed
            action = self._get_action_from_speed(updated_speed)

        # computign next positions
        _, cur_pos = super()._move_shape(
            speed=updated_speed,
            cur_pos=cur_pos,
            img_size=img_size,
            digit_size=digit_size
        )
        
        return updated_speed, cur_pos, action