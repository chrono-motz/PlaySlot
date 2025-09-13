"""
MetaWorld ButtonPress Dataset, modified to load a HuggingFace LeRobot dataset
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np

from lib.logger import print_
from datasets import load_dataset
from datasets.features.image import Image as HFImage

# Define action projection sizes
ACTION_INPUT_SIZE = 14
ACTION_OUTPUT_SIZE = 4

class BaseButtonPress(Dataset):
    """
    Base Button-Press Dataset, modified to load LeRobot datasets.
    """

    def __init__(self, dataset_name="lerobot/aloha_sim_transfer_cube_human", split="train",
                 num_frames=20, sample_rate=1, img_size=(64, 64), **kwargs):
        """
        Dataset Initializer
        """
        # --- Start of Changes ---

        # 1. Re-enable validation of the 'split' argument
        if split not in ["train", "val", "valid", "eval", "test"]:
            raise ValueError(f"Unknown dataset split {split}...")
        split = "val" if split in ["val", "valid", "validation"] else split
        split = "test" if split in ["test", "eval"] else split
        # The line `split = "train"` has been removed.

        self.dataset_name = dataset_name
        self.split = split
        self.num_frames = num_frames
        self.img_size = img_size
        self.sample_rate = sample_rate
        self.random_start = True if split == "train" else False
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
        self.resize = transforms.Resize(img_size)

        # Create the action projection layer
        self.action_projector = nn.Sequential(
            nn.Linear(ACTION_INPUT_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_OUTPUT_SIZE)
        )

        # 2. Always load the 'train' split from Hugging Face, as it's the only one available
        self.hf_dataset = load_dataset("lerobot/aloha_sim_transfer_cube_human_image", split="train")
        
        # Map `frame_index` to dataset index for sampling
        self.frames_to_indices = {}
        for idx, frame_index in enumerate(self.hf_dataset["frame_index"]):
            episode_index = self.hf_dataset["episode_index"][idx]
            if episode_index not in self.frames_to_indices:
                self.frames_to_indices[episode_index] = []
            self.frames_to_indices[episode_index].append(idx)

        all_episodes = sorted(list(self.frames_to_indices.keys()))

        # 3. Manually partition the full list of episodes into train and val sets
        split_point = int(len(all_episodes) * 0.95)  # 95% for training, 5% for validation
        if self.split == "train":
            self.episodes = all_episodes[:split_point]
        else: # Handles 'val' or 'test'
            self.episodes = all_episodes[split_point:]

        # --- End of Changes ---

    def __len__(self):
        """ Number of episodes in the dataset """
        return len(self.episodes)

    def __getitem__(self, index):
        """ Fetching a sequence from the dataset """
        episode_index = self.episodes[index]
        episode_indices = self.frames_to_indices[episode_index]

        # Get a random start index for the sequence
        start_idx = 0
        if self.random_start:
            # Ensure there's enough room for a full sequence
            max_start_idx = len(episode_indices) - self.num_frames
            if max_start_idx > 0:
                start_idx = np.random.randint(0, max_start_idx + 1)
            # If not enough frames, start_idx remains 0
        
        # Sample frames for the sequence
        sequence_indices = episode_indices[start_idx : start_idx + self.num_frames]
        
        # Load the data for the sequence
        data_batch = self.hf_dataset.select(sequence_indices)

        # Extract and process images
        imgs = []
        for img_data in data_batch["observation.images.top"]:
            if isinstance(img_data, dict):
                img = self.to_pil(img_data["pixels"])
            elif isinstance(img_data, HFImage):
                img = img_data.convert("RGB")
            else:
                img = img_data
            img = self.resize(img)
            imgs.append(self.to_tensor(img))
        imgs = torch.stack(imgs)

        # Extract and project actions
        actions = torch.tensor(data_batch["action"], dtype=torch.float32)
        # Apply the projection layer and detach the tensor
        actions = self.action_projector(actions).detach()

        # Pad actions if the sequence length is shorter than num_frames
        if actions.shape[0] < self.num_frames:
            actions_to_add = self.num_frames - actions.shape[0]
            actions = torch.cat([actions, actions[-1:].repeat(actions_to_add, 1)])
        
        targets = imgs
        all_reps = {
            "actions": actions,
        }
        return imgs, targets, all_reps

    def get_num_frames_per_episode(self):
        """ Getting the number of imgs in each episode to compute a histogram """
        num_imgs = [len(self.frames_to_indices[e]) for e in self.episodes]
        return num_imgs


class ButtonPress(BaseButtonPress):
    """
    ButtonPress dataset for the video prediction model
    """
    def __init__(self, **kwargs):
        super(ButtonPress, self).__init__(**kwargs)

class ButtonPressExpertDemos(BaseButtonPress):
    """
    ButtonPressExpertDemos dataset for the video prediction model
    """
    def __init__(self, **kwargs):
        super(ButtonPressExpertDemos, self).__init__(**kwargs)