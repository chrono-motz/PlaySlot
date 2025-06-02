"""
DataClass for the Sketchy dataset.

In this dataset, a robotic gripper grabs and moves around some colorful cubes
  --> https://sites.google.com/view/data-driven-robotics/
We follow the preprocessing steps from GENESIS-v2

About the Actions:
  - Action vectors have 7 variables:
  - Six continuous degrees of freedom:
    - 3 Cartesian translational targets of the gripper pinch point
    - 3 rotational velocity targets of the gripper pinch point
  - 1 binary control of gripper fingers
"""

import os
from glob import glob
import numpy as np
from tqdm import tqdm
import imageio
import torch
from torch.utils.data import Dataset
from torchvision import transforms



class Sketchy(Dataset):
    """
    DataClass for the Sketchy dataset.
    In this dataset, a robotic gripper grabs and moves around some colorful cubes.
    We follow the preprocessing steps from GENESIS-v2

    Args:
    -----
    split: string
        Dataset split to load
    num_frames: int
        Desired length of the sequences to load
    seq_step: int
        Temporal resolution at which we use frames.
        seq_step=2 means we use one frame out of each two
    max_overlap: float
        Determines amount of overlap between consecuitve sequences,
        given as percentage of num_frames.
        For instance, 0.25 means that consecutive sequence will overlap
        for 25% of the frames
    img_size: tuple
        Images are resized to this resolution
    """

    DATAPATH = "/home/nfs/inf6/data/datasets/SketchyRobot/new_data/sketchy_data"
    SPLIT_IDX = [325, 360]  # approx 80% training, 10% validation and 10% test

    CAMERAS = ["front_right", "front_left"]
    MODES = ["all", "full_only", "crop_only"]  # images can be full, crops or both

    ACTIONS = [  # This could be off
            "X-velocity", "Y-velocity", "Z-velocity",
            "", "", "",  # Euler angles?
            "Gripper Control (binary)"
        ]

    NICE_SIZES = [(128, 128), (64, 64), (96, 96)]  # (600 x 960) resized to (128x128)
    NUM_FRAMES_LIMITS = {
            "train": [10, 100],
            "val": [30, 60],
            "test": [30, 60],  # please keep this fixed
        }


    def __init__(self, datapath, split, num_frames, seq_step=2, img_size=(128, 128),
                 max_overlap=0., mode="all", **kwargs):
        """ Dataset initializer"""
        assert mode in Sketchy.MODES, f"Unknown {mode = }. Use on of {Sketchy.MODES}..."
        assert split in ["train", "val", "valid", "test"]
        
        self.is_custom = split in ["val", "valid", "test"]
        split = "val" if split in ["valid", "test"] else split
        assert max_overlap <= 0.95 and max_overlap >= 0
        self.data_dir = datapath
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Sketchy dataset does not exist in {self.data_dir}...")

        if split != "train":
            print("Setting Sketchy mode to Full-Only!")
            mode = "full_only"
        
        # dataset parameters
        self.split = split
        self.given_num_frames = num_frames
        self.num_frames = self._check_num_frames_param(split=split, num_frames=num_frames)
        self.seq_step = seq_step
        self.img_size = img_size
        self.max_overlap = max_overlap
        self.mode = mode

        # aux modules
        self.resizer = transforms.Resize(self.img_size)

        # generating sequences
        self.allow_seq_overlap = (split == "train")
        self.episode_data = self._get_episode_data()
        self.episode_names = list(self.episode_data.keys())
        self.valid_sequences, self.valid_sequence_keys = self._find_valid_sequences()
        
        if self.is_custom:
            new_valid_sequence_keys = []
            new_valid_sequences = {}
            for ep_name in self.episode_names:
                new_valid_sequence_keys.append(f"{ep_name}/fl_full")
                new_valid_sequences[ep_name] = {}
                new_valid_sequences[ep_name]["fl_full"] = self.valid_sequences[f"{ep_name}_0"]["fl_full"]
                new_valid_sequences[ep_name]["actions"] = self.valid_sequences[f"{ep_name}_0"]["actions"]
            self.valid_sequences = new_valid_sequences
            self.valid_sequence_keys = new_valid_sequence_keys
        return

    def __len__(self):
        """ Number of sequences in dataset """
        return len(self.valid_sequence_keys)

    def __getitem__(self, i):
        """ Sampling a sequence from the dataset """
        cur_key = self.valid_sequence_keys[i]
        episode, seq = cur_key.split("/")

        # getting frame paths and actions
        cur_frames = self.valid_sequences[episode][seq]
        cur_frames = cur_frames[:self.given_num_frames]
        actions = self.valid_sequences[episode]["actions"]
        actions = actions[:self.given_num_frames][:, 0]

        # loading imgs
        imgs = [imageio.imread(frame) / 255. for frame in cur_frames]
        imgs = np.stack(imgs, axis=0)
        imgs = torch.Tensor(imgs).permute(0, 3, 1, 2)
        imgs = self.resizer(imgs)

        data = {
                "actions": torch.tensor(actions),
                "episode": episode,
                "seq_type": seq,
                "frame_names": cur_frames
            }
        return imgs, imgs, data


    def _find_valid_sequences(self):
        """
        Finding valid sequences in the corresponding dataset split.
        """
        print(f"Preparing Sketchy {self.split} set...")

        # stats about episodes and sequence lengths
        episode_lengths = [ep["seq_len"] for _, ep in self.episode_data.items()]
        accumulated_sum = np.cumsum(episode_lengths)
        total_num_frames = sum(episode_lengths)
        seq_len = (self.num_frames - 1) * self.seq_step + 1

        last_valid_idx = -1 * seq_len
        last_episode = ""

        valid_sequences = {}
        valid_seq_keys = []
        for idx in tqdm(range(total_num_frames - seq_len + 1)):
            # handle overlapping sequences
            episode, start_frame_idx = self._get_episode_from_idx(idx, accumulated_sum)

            # no overlap and currently in overlapping idx
            if (not self.allow_seq_overlap and
                (idx < last_valid_idx + seq_len) and
                (last_episode == episode)):
                continue
            # allowed overlap, but not yet reached the limit
            if (self.allow_seq_overlap and 
                (idx < last_valid_idx + seq_len * (1 - self.max_overlap))):
                continue

            # obtaining frames for the sequence, if current idx is valid
            cur_valid_seqs = self._frames_from_id(idx, episode, start_frame_idx)
            if len(cur_valid_seqs) == 0:
                continue

            # saving current sequences
            idx_ = idx - start_frame_idx
            valid_sequences = {**valid_sequences, **cur_valid_seqs}
            valid_seq_keys += [f"{episode}_{idx_}/{k}" for k in cur_valid_seqs[f"{episode}_{idx_}"].keys()]
            last_valid_idx = idx
            last_episode = episode

        # removing actions from valid_seq_keys
        valid_seq_keys = [k for k in valid_seq_keys if "actions" not in k]

        if len(valid_sequences) <= 0:
            raise ValueError("No valid sequences were found...")
        if len(valid_seq_keys) <= 0:
            raise ValueError("No valid sequence keys were found...")
        return valid_sequences, valid_seq_keys


    def _get_episode_from_idx(self, frame_idx, accumulated_sum):
        """ Getting episode name from idx """
        start_indices = [0] + accumulated_sum.tolist()
        idx = np.argwhere(accumulated_sum > frame_idx)[0][0]
        episode = self.episode_names[idx]
        start_frame_idx = start_indices[idx]
        return episode, start_frame_idx


    def _frames_from_id(self, frame_idx, episode, start_frame_idx):
        """ Fetching frame paths given frame_idx """

        # getting frame indices for the sequence
        start_offset = frame_idx - start_frame_idx
        end_offset = start_offset + self.num_frames * self.seq_step
        frame_offsets = range(start_offset, end_offset, self.seq_step)

        # making sure the sequence has enough length
        if end_offset > len(self.episode_data[episode]["fr_full"]):
            return {}

        # paths for all imgs
        cur_key = f"{episode}_{start_offset}"
        valid_seqs = {cur_key: {}}
        for k, v in self.episode_data[episode].items():
            if k == "actions":
                valid_seqs[cur_key]["actions"] = self.episode_data[episode]["actions"][frame_offsets]
            if self.mode in ["crop_only", "all"] and "crop" in k:
                valid_seqs[cur_key][k] = [v[f] for f in frame_offsets]
            elif self.mode in ["full_only", "all"] and "full" in k:
                valid_seqs[cur_key][k] = [v[f] for f in frame_offsets]
            else:
                continue
        return valid_seqs


    def _ep_num_from_id(self, frame_idx):
        """ Getting episode number given frame_idx """
        episode_idx = np.argwhere(self.accumulated_sum > frame_idx)[0][0]
        episode_dir = f"episode_{episode_idx:04d}"
        return episode_dir


    def _check_num_frames_param(self, num_frames, split):
        """
        Making sure the given 'num_frames' is valid for the corresponding split
        """
        if num_frames < self.NUM_FRAMES_LIMITS[split][0] or num_frames > self.NUM_FRAMES_LIMITS[split][1]:
            N = self.NUM_FRAMES_LIMITS[split][0]
            print(f"Sketchy {split} sequences must have at least {N} frames.")
            print(f"  --> Your {num_frames = } will be overridden to {N}...")
            num_frames = N
        return num_frames


    def _get_episode_data(self):
        """ Obtaining paths for the frames and episodes for the current dataset split """
        print("Loading Sketchy data into data-structure...")
        all_episodes = sorted(os.listdir(self.data_dir))
        if self.split == "train":
            episodes = all_episodes[:self.SPLIT_IDX[0]]
        elif self.split == "val":
            episodes = all_episodes[self.SPLIT_IDX[0]:self.SPLIT_IDX[1]]
        else:
            episodes = all_episodes[self.SPLIT_IDX[1]:]

        # Loading actions and img-paths into data structure
        data = {}
        for i, ep in enumerate(tqdm(episodes)):
            data[ep] = {}
            cur_path = os.path.join(self.data_dir, ep)
            data[ep]["fl_full"] = sorted(glob(os.path.join(cur_path, "fl_*_full.png")))
            data[ep]["fr_full"] = sorted(glob(os.path.join(cur_path, "fr_*_full.png")))
            for i in range(6):
                data[ep][f"fl_crop_{i}"] = sorted(glob(os.path.join(cur_path, f"fl_*_crop_{i}.png")))
                data[ep][f"fr_crop_{i}"] = sorted(glob(os.path.join(cur_path, f"fr_*_crop_{i}.png")))
            data[ep]["actions"] = np.load(os.path.join(cur_path, "actions.npy"))
            data[ep]["seq_len"] = len(data[ep]["fl_full"])
        return data


