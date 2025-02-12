"""
Dataset class to load the BlockPush dataset, either for the 
Random-Exploration or for the Exper-Demos variants.
"""

import glob
import os
import os.path as osp
import torch.nn.functional
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from lib.logger import print_



class BaseBlockPush(Dataset):
    """ 
    Base class used to abstrsact functionality from the different BlockPush variants.
    """

    def __init__(self, split, dataset_name, datapath, ep_len=25, num_frames=20,
                 random_start=True, img_size=(64, 64)):
        """ Dataset Initializer """
        if split not in ["train", "val", "valid", "eval", "test"]:
            raise ValueError(f"Unknown dataset split {split}...")
        split = "val" if split in ["val", "valid"] else split
        split = "test" if split in ["test", "eval"] else split

        self.dataset_name = dataset_name
        self.datapath = datapath
        self.split = split
        self.ep_len = ep_len
        self.num_frames = num_frames
        self.random_start = random_start
        self.img_size = img_size
        self.to_tensor = transforms.ToTensor()

        # loading data
        self._get_data()

        print_("Instanciating ButtonPress dataset:")
        print_(f"  --> datapath: {self.datapath}")
        print_(f"  --> split: {self.split}")
        print_(f"  --> NumFrames: {self.num_frames}")
        print_(f"  --> Random Start: {self.random_start}")
        print_(f"  --> Seqs Per Episode: {self.seq_per_episode}")
        return
    
    def __len__(self):
        """ Number of episodes in the dataset """
        length = len(self.episodes)
        return length

    def __getitem__(self, index):
        """
        Loading a sequence
        """
        # Implement continuous indexing
        ep = index // self.seq_per_episode
        offset = index % self.seq_per_episode
        end = offset + self.num_frames

        # loading images and segmentations for the corresponding episod
        imgs = []
        e = self.episodes[ep]
        for image_index in range(offset, end):
            image_path = osp.join(e[image_index])
            imgs.append(self._load_img(image_path))

        imgs = torch.stack(imgs, dim=0).float()
        actions = self._load_actions(episode_idx=ep, offset=offset, end=end)
        all_reps = {
            "videos": imgs,
            "actions": actions,
            "masks": torch.tensor([]),
        }
        return imgs, imgs, all_reps

    def _load_img(self, img_path):
        """ Loading image """
        img = Image.open(img_path)
        img = img.resize(self.img_size)
        img = self.to_tensor(img)[:3]
        return img

    def _load_actions(self, episode_idx, offset, end):
        """ Loading actions for the current sequence """
        base_path = "/" + osp.join(*(self.episodes[episode_idx][0].split("/")[:-1]))
        if self.episodes[episode_idx][0].startswith("."):
            base_path = base_path[1:]  # removing initial '/'
        action_path = osp.join(base_path, "actions.npy")
        actions = torch.from_numpy(np.load(action_path)[offset:end])
        return actions
    
    def _get_data(self):
        """
        Creating a data structure with img paths and idxs to simplify data loading
        """
        self.folders = []
        for file in os.listdir(self.root):
            try:
                self.folders.append(int(file))
            except ValueError:
                continue
        self.folders.sort()        
        self._get_data_base()
        return            

    def _get_data_base(self):
        """ Loading images from data directories and assembling then into episodes """
        self.episodes = []
        self.EP_LEN = self.ep_len
        if self.split == "train" and self.random_start:
            self.seq_per_episode = self.EP_LEN - self.num_frames + 1
        else:
            self.seq_per_episode = 1
        
        paths_template = None
        for i, f in enumerate(tqdm(self.folders)):
            dir_name = os.path.join(self.root, str(f))
            if i == 0:
                paths = list(glob.glob(osp.join(dir_name, '*.png')))
                self.has_segmentations = any(["seg_" in p.split("/")[-1] for p in paths])
                paths = [p for p in paths if p.split("/")[-1].split(".")[0].isdigit()]
                get_num = lambda x: int(osp.splitext(osp.basename(x))[0])
                paths.sort(key=get_num)
                paths_template = [p.split("/")[-1] for p in paths]
            else:
                paths = [os.path.join(dir_name, p) for p in paths_template]
            self.episodes.append(paths)   
        return



class BlockPush(BaseBlockPush):
    """
    DataClass for loading sequences from the BlockPush dataset

    During training, we can sample a random subset of frames in the episode.
    At inference time, we always start from the first frame.

    Args:
    -----
    split: string
        Dataset split to load. Can be one of ['train', 'val', 'test']
    datapath: string
        Path where the dataset is stored
    ep_len: int
        Number of frames in an episode. Default is 30
    sample_length: int
        Number of frames in the sequences to load
    random_start: bool
        If True, first frame of the sequence is sampled at random.
        Otherwise, starting frame is always the first frame in the sequence.
    """
    
    def __init__(self, split, dataset_name, datapath, ep_len=25, num_frames=20,
                 random_start=True, img_size=(64, 64), **kwargs):
        """
        Dataset Initializer
        """
        split = "val" if split in ["val", "valid"] else split
        split = "test" if split in ["test", "eval"] else split
        self.root = osp.join(datapath, split)
        super().__init__(
            split=split,
            dataset_name=dataset_name,
            datapath=datapath,
            ep_len=ep_len,
            num_frames=num_frames,
            random_start=random_start,
            img_size=img_size
        )
        return
    


class BlockPushExpertDemos(BaseBlockPush):
    """
    Module for loading the Expert Demonstrations from the Block-Push dataset
    """
    
    def __init__(self, split, dataset_name, datapath, ep_len=25, num_frames=20,
                 random_start=True, img_size=(64, 64), num_expert_demos=None, **kwargs):
        """
        Dataset Initializer
        """
        self.root = osp.join(datapath)
        if random_start is True:
            print_(f"WARNING! Overriding 'random_start' to False in Expert-BlockPush")
        super().__init__(
                split=split,
                dataset_name=dataset_name,
                datapath=datapath,
                ep_len=ep_len,
                num_frames=num_frames,
                random_start=False,
                img_size=img_size
            )
        
        # keeping only the specified number of expert demonstrations
        self.num_expert_demos = num_expert_demos
        assert num_expert_demos is not None, "'num_expert_demos' must be specified..."
        if num_expert_demos > 0 and self.split == "train":
            print_(f"  --> Keeping only the first {num_expert_demos} expert demonstrations")
            self.episodes = self.episodes[:num_expert_demos]
        return
    
    def _get_data(self):
        """ Overriding the '_get_data' function to enforce custom splits """
        super()._get_data()
        
        # train-test splits
        num_episodes = len(self.episodes)
        thr_ep = int(0.9 * num_episodes)
        if self.split == "train":
            self.episodes = self.episodes[:thr_ep]
        elif self.split in ["val", "test"]:
            self.episodes = self.episodes[thr_ep:]
        else:
            raise ValueError(f"{self.split = } was not recognized...")
        return
        
