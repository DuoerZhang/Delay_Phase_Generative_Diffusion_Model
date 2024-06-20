
import os
import warnings
import os.path as osp
import numpy as np
import PIL.Image
from PIL import Image
from PIL import ImageFile
import SimpleITK as sitk
import json
import torch
import torch.nn.functional as F
from einops import rearrange
from torchvision import transforms
from torchvision.datasets.video_utils import VideoClips
import random
import pickle
from scipy.ndimage import zoom

from torchvision.datasets import UCF101
from torchvision.datasets.folder import make_dataset
from torch.utils.data import Dataset
from abc import ABC, abstractmethod

class BaseDataset(Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        pass

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass

class MPIDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self,root,transform = None):
        """Initialize this dataset class.A

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self)
        self.root = root
        self.ids = os.listdir(self.root)
        self.ids = list(self.ids)
        self.transform = transform
    def _load_nii_from_path(self, path):
        # load nii into tensor without change value
        img = sitk.GetArrayFromImage(sitk.ReadImage(path))
        img = torch.from_numpy(np.float32(img))
        img = img.unsqueeze(0)
        return img
    def _load_nii_from_path_zoom(self, path):
        # load nii into tensor without change value
        img = sitk.GetArrayFromImage(sitk.ReadImage(path))
        img = torch.from_numpy(np.float32(img))
        img = img.unsqueeze(0)
        return img
    def resize_crop(image, resize_factor, im_flag):
        assert(im_flag=='i'or im_flag=='m')
        # resize_factor = target_shape/np.array(image.shape)
        # # z轴方向不做resample
        # resize_factor[0] = 1.
        if im_flag == 'i':
            newVol = zoom(image, resize_factor, mode='nearest', order=2)
        else:
            newVol = zoom(image,resize_factor,mode='constant',order=0)
        # centercrop 256
            
        return newVol
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        id_ = self.ids[index]

        path_0 = self.root + '/' + id_ + '/input_p.nii'# 平扫
        path_1 = self.root + '/' + id_ + '/input_d.nii'# 动脉期
        path_2 = self.root + '/' + id_ + '/input_j.nii'# 静脉期
        path_3 = self.root + '/' + id_ + '/input_m.nii'# 门脉期
        path_mask = self.root + '/' + id_ + '/mask_m.nii'# 静脉期、tumor
        path_liver = self.root + '/' + id_ + '/liver_m.nii'# 静脉期、liver
        img0 = self._load_nii_from_path(path_0)
        img1 = self._load_nii_from_path(path_1)
        img2 = self._load_nii_from_path(path_2)
        img3 = self._load_nii_from_path(path_3)
        
        mask2 = sitk.GetArrayFromImage(sitk.ReadImage(path_mask))
        mask2 = torch.from_numpy(np.float32(mask2))
        mask2 = mask2.unsqueeze(0)
        liver2 = sitk.GetArrayFromImage(sitk.ReadImage(path_liver))
        liver2 = torch.from_numpy(np.float32(liver2))
        liver2 = liver2.unsqueeze(0)
        # print(img0.shape)
        img0 = img0 * 2. - 1.
        img1 = img1 * 2. - 1.
        img2 = img2 * 2. - 1.
        img3 = img3 * 2. - 1.
        mask2 = mask2 *2 - 1.
        liver2 = liver2 *2.-1.
        return img0, img1, img2, img3, id_, mask2,liver2
    def __len__(self):
        return (len(self.ids))