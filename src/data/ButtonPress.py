""" 
MetaWorld ButtonPress Dataset
"""

import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from lib.logger import print_



class BaseButtonPress(Dataset):
    """
    Base Button-Press Dataset
    """

    def __init__(self, split, num_frames=20, sample_rate=1,
                 random_start=True, img_size=(64, 64), **kwargs):
        """
        Dataset Initializer
        """
        if split not in ["train", "val", "valid", "eval", "test"]:
            raise ValueError(f"Unknown dataset split {split}...")
        split = "val" if split in ["val", "valid", "validation"] else split
        split = "test" if split in ["test", "eval"] else split

        self.split = split
        self.num_frames = num_frames
        self.random_start = random_start
        self.img_size = img_size
        self.sample_rate = sample_rate
        self.random_start = random_start if split == "train" else False
        self.to_tensor = transforms.ToTensor()
        return

    def __len__(self):
        """ Number of episodes in the dataset """
        length = len(self.episodes)
        return length

    def __getitem__(self, index):
        """ Fetching a sequence from the dataset """
        episode = self.episodes[index]
        img_names = self.get_imgs_names_to_load(episode)
        num_imgs = len(img_names)
        if num_imgs < self.num_frames * self.sample_rate + 1:
            return self.__getitem__(np.random.randint(0, self.num_episodes))

        # loading images
        if self.random_start:
            start_idx = np.random.randint(0, num_imgs - self.num_frames * self.sample_rate)
        else:
            start_idx = 0
        end_idx = start_idx + self.num_frames * self.sample_rate
        img_names = img_names[start_idx:end_idx:self.sample_rate]
        imgs = self.load_imgs(episode, img_names)

        # load actions
        actions = torch.from_numpy(
                np.load(
                    os.path.join(episode, "actions.npy")
                )[start_idx:end_idx:self.sample_rate]
            )

        targets = imgs
        all_reps = {
            "videos": imgs,
            "episode": episode,
            "actions": actions,
            "img_paths": img_names
        }
        return imgs, targets, all_reps


    def print_db(self):
        """ Printing DB stats and parameters """
        print_("Instanciating MetaWorld dataset:")
        print_(f"  --> datapath: {self.root}")
        print_(f"  --> split: {self.split}")
        print_(f"  --> NumFrames: {self.num_frames}")
        print_(f"  --> Sample Rate: {self.sample_rate}")
        print_(f"  --> Random Start: {self.random_start}")
        print_(f"  --> Img Size: {self.img_size}")
        return


    def load_imgs(self, episode, img_names):
        """ Loading images as torch tensors """
        imgs = []
        for img_name in img_names:
            cur_p = os.path.join(episode, img_name)
            img = Image.open(cur_p)
            img = img.resize(self.img_size)
            img = self.to_tensor(img)[:3]
            imgs.append(img)                                               
        imgs = torch.stack(imgs, dim=0).float()
        return imgs



class ButtonPress(BaseButtonPress):
    """
    ButtonPress Dataset
    """
    
    def __init__(self, split, datapath, num_frames=20, sample_rate=1,
                 random_start=True, img_size=(64, 64), **kwargs):
        """ Dataset Initializer """
        super().__init__(
                split=split,
                num_frames=num_frames,
                sample_rate=sample_rate,
                random_start=random_start,
                img_size=img_size,
        )

        # getting data path
        self.root = os.path.join(datapath, self.split)
        if not os.path.exists(self.root):
            raise FileNotFoundError(f"Data path {self.root} does not exist...")

        # Get all episodes
        cur_episodes = sorted([int(f) for f in os.listdir(self.root)])
        self.episodes = [os.path.join(self.root, str(f)) for f in cur_episodes]
        
        self.print_db()
        return
    
    def get_imgs_names_to_load(self, episode):
        """ Fetching image paths to load"""
        img_names = sorted(
                [f for f in os.listdir(episode) if f.endswith(".png")],
                key=lambda f: int(f.split(".")[0])
            )
        num_imgs = len(img_names)
        if num_imgs < self.num_frames * self.sample_rate:
            raise ValueError(f"{num_imgs = } must be > {self.num_frames * self.sample_rate}")
        return img_names



class ButtonPressExpertDemos(BaseButtonPress):
    """ 
    Custom ButtonPress dataset generated using expert policies.
    This module can be used for imitation-learning/behavior cloning.
    """

    def __init__(self, split, datapath, num_frames=20, sample_rate=1, random_start=True,
                 num_expert_demos=None, img_size=(64, 64), **kwargs):
        """ Dataset Initializer """
        super().__init__(
                split=split,
                num_frames=num_frames,
                sample_rate=sample_rate,
                random_start=random_start,
                img_size=img_size,
        )
        self.root = datapath
        self.num_expert_demos = num_expert_demos
        assert num_expert_demos is not None, "'num_expert_demos' must be specified..."

        # Get all episodes and enforce train/test split
        episodes = sorted(os.listdir(self.root))
        num_eps = len(episodes)
        thr = int(0.9 * num_eps)
        if self.split == "train":
            episodes = episodes[:thr]
        else:
            episodes = episodes[thr:]
        self.episodes = [os.path.join(self.root, ep) for ep in episodes]
    
        # keeping only the desired number of training episodes
        if self.split == "train":
            num_episodes = len(self.episodes)
            if num_episodes >= num_expert_demos:
                print_(f"Metaworld ButtonPress with Expert Policies:")
                print_(f"  --> Keeping {num_expert_demos}/{num_episodes} episodes...")
                self.episodes = self.episodes[:num_expert_demos]
            else:
                raise ValueError(f"{num_episodes = } cannot be < {num_expert_demos = }")
        self.num_episodes = len(self.episodes)
        self.print_db()
        return

    def get_imgs_names_to_load(self, episode):
        """ Fetching image paths to load"""
        img_names = sorted([f for f in os.listdir(episode) if f.endswith(".png")])
        return img_names

    def __getitem__(self, index):
        """
        Fetching a sequence from the dataset
        This requires overriding the Base __getitem__ as 
        """
        episode = self.episodes[index]
        img_names = self.get_imgs_names_to_load(episode)
        num_imgs = len(img_names)

        # loading images
        imgs = self.load_imgs(episode, img_names)

        # repeating the final image to match 'num_frames' or sampling image subset
        frames_to_add = self.num_frames - num_imgs
        if frames_to_add > 0:
            extra_imgs = imgs[-1:].repeat(frames_to_add, 1, 1, 1)
            imgs = torch.cat([imgs, extra_imgs], dim=0)
        elif frames_to_add < 0:
            if self.random_start:
                start_idx= np.random.randint(0, num_imgs - self.num_frames)
            else:
                start_idx = 0
            end_idx = start_idx + self.num_frames
            imgs = imgs[start_idx:end_idx]
        else:
            start_idx = 0
            end_idx = self.num_frames
            imgs = imgs[start_idx:end_idx]
                            
        # load actions and pad/remove
        actions = torch.from_numpy(
                np.load(
                    os.path.join(episode, "actions.npy")
                )
            )
        if frames_to_add > 0:
            actions = torch.cat([actions, actions[-1:].repeat(frames_to_add, 1)])
        else:
            actions = actions[start_idx:end_idx-1]
        actions = actions.clamp(-1, 1)
        targets = imgs
        all_reps = {
            "actions": actions,
        }
        return imgs, targets, all_reps


    def get_num_frames_per_episode(self):
        """ Getting the number of imgs in each episode to compute a histogram """
        num_imgs = []
        for i in tqdm(range(len(self))):
            episode = self.episodes[i]
            img_names = self.get_imgs_names_to_load(episode)
            num_imgs.append(len(img_names))
        return num_imgs



