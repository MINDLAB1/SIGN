import os
from torch.utils.data import Dataset as dataset_torch
import numpy as np
import random
import SimpleITK as sitk
from parameters import *


def _make_dataset(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            # if is_image_file(fname):
            path = os.path.join(root, fname)
            item = path
            images.append(item)
    return images


def _make_image_namelist(dir):
    images = []
    namelist = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if fname.endswith('t1.nii'):
                item_name = fname
                namelist.append(item_name)
                item_path = os.path.join(root, fname)
                images.append(item_path)
    return images, namelist


def mean_std_normalization(img, epsilon=1e-8):
    img_mean = np.mean(img)
    img_std = np.std(img) + epsilon
    img = (img - img_mean) / img_std
    return img


class data_set(dataset_torch):
    def __init__(self, root, split='train', data_type='BraTS'):
        self.root = root
        assert split in ('train', 'val', 'test')
        assert data_type in ('BraTS', 'SISS')
        self.split = split
        self.data_type = data_type
        self.imgs, self.nlist = _make_image_namelist(os.path.join(self.root, self.split + '_' + self.data_type))

        self.epi = 0
        self.img_num = len(self.imgs)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path_t1 = self.imgs[index]
        case_name = self.nlist[index]
        path_mask = path_t1.replace('t1', 'annot')
        path_t2 = path_t1.replace('t1', 't2')
        path_flair = path_t1.replace('t1', 'flair')

        img_t1 = sitk.ReadImage(path_t1, sitk.sitkInt16)
        img_t2 = sitk.ReadImage(path_t2, sitk.sitkInt16)
        img_flair = sitk.ReadImage(path_flair, sitk.sitkInt16)
        mask = sitk.ReadImage(path_mask, sitk.sitkInt16)

        img_t1_array = sitk.GetArrayFromImage(img_t1)
        img_t2_array = sitk.GetArrayFromImage(img_t2)
        img_flair_array = sitk.GetArrayFromImage(img_flair)
        mask_array = sitk.GetArrayFromImage(mask)

        img_t1_array = img_t1_array.astype(np.float32)
        img_t1_array = mean_std_normalization(img_t1_array)
        img_t2_array = img_t2_array.astype(np.float32)
        img_t2_array = mean_std_normalization(img_t2_array)
        img_flair_array = img_flair_array.astype(np.float32)
        img_flair_array = mean_std_normalization(img_flair_array)

        img_t1_list = np.expand_dims(img_t1_array, axis=0)
        img_t2_list = np.expand_dims(img_t2_array, axis=0)
        img_flair_list = np.expand_dims(img_flair_array, axis=0)
        mask_list = np.expand_dims(mask_array, axis=0)

        img_t1_list = np.array(img_t1_list)
        img_t2_list = np.array(img_t2_list)
        img_flair_list = np.array(img_flair_list)
        mask_list = np.array(mask_list)

        return img_t1_list, img_t2_list, img_flair_list, mask_list, case_name.replace('t1', 'predict')
