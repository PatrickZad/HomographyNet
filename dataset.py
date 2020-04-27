import os
import torch
from torch.utils.data import Dataset
import cv2
from common import *

proj_dir = os.getcwd()
commom_dir = os.path.dirname(proj_dir)
data_dir = os.path.join(commom_dir, 'Datasets')


class AerialTrain(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


class AerialVal(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


class ResiscDataset(Dataset):
    # 128*128 patches
    def __init__(self, file_path):
        self.common_path = os.path.join(data_dir, 'NWPU-RESISC45')
        self.file_paths = file_path
        self.patch_size = (128, 128)
        self.scale_extent = (0.75, 1.5)
        self.purtube = 32
        self.erasing_prob = 0.5

    def __getitem__(self, idx):
        img_array = cv2.imread(os.path.join(self.common_path, self.file_paths[idx]))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        # random scale
        scale_factor = rand(self.scale_extent[0], self.scale_extent[1])
        scaled_img = cv2.resize(img_array, (0, 0), fx=scale_factor, fy=scale_factor)
        # random patch
        h, w = scaled_img.shape[0], scaled_img.shape[1]
        x_limit = (self.purtube, w - self.purtube - 1 - self.patch_size[0] + 1)  # left-top
        corner_x = np.random.randint(x_limit[0], x_limit[1])
        y_limit = (self.purtube, h - self.purtube - 1 - self.patch_size[1] + 1)
        corner_y = np.random.randint(y_limit[0], y_limit[1])
        # random homography
        cornersA = np.array([[corner_x, corner_y], [corner_x + self.patch_size[0] - 1, corner_y],
                             [corner_x + self.patch_size[0] - 1, corner_y + self.patch_size[1] - 1],
                             corner_x, corner_y + self.patch_size[1] - 1])
        h_ab_4p = rand(-self.purtube, self.purtube + 1, size=(4, 2))
        cornersB = cornersA + h_ab_4p
        h_ab = cv2.getPerspectiveTransform(cornersA, cornersB)
        h_ba = np.linalg.inv(h_ab)
        patchA = scaled_img[corner_y:corner_y + self.patch_size[1], corner_x:corner_x + self.patch_size[0], :].copy()
        patchB = warpcrop_in_same_coordsys(scaled_img, h_ba,
                                           (corner_x, corner_y, self.patch_size[0], self.patch_size[1]))
        # color distortion
        patchA = corlor_distortion(patchA)
        patchA = np.float32(cv2.cvtColor(patchA, cv2.COLOR_RGB2GRAY)) / 255
        patchB = np.float32(cv2.cvtColor(patchB, cv2.COLOR_RGB2GRAY)) / 255
        # random erasing
        patchA = random_erasing(patchA, self.erasing_prob)
        patchB = random_erasing(patchB, self.erasing_prob)
        patchA = patchA.transpose((2, 0, 1))
        patchB = patchB.transpose((2, 0, 1))
        data_array = np.concatenate([patchA, patchB], axis=0)
        return torch.tensor(data_array), torch.tensor(h_ab_4p).reshape((4 * 2,))

    def __len__(self):
        return len(self.file_paths)


def getAerialData():
    pass


def getResiscData(train_proportion=0.7):
    data_base = os.path.join(data_dir, 'NWPU-RESISC45')
    scenes = os.listdir(data_base)
    indices = np.arange(0, 700)
    np.random.shuffle(indices)
    train_files = []
    val_files = []
    for scene in scenes:
        img_files = os.listdir(os.path.join(data_base, scene))
        for i in range(int(indices.shape[0] * train_proportion)):
            img_file = img_files[indices[i]]
            train_files.append(os.path.join(scene, img_file))
        for i in range(int(indices.shape[0] * train_proportion), indices.shape[0]):
            img_file = img_files[indices[i]]
            val_files.append(os.path.join(scene, img_file))
    return ResiscDataset(train_files), ResiscDataset(val_files)
