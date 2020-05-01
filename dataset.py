import os
import torch
from torch.utils.data import Dataset
import cv2
from common import *
from skimage import io, transform, color, util

proj_dir = os.getcwd()
commom_dir = os.path.dirname(proj_dir)
data_dir = os.path.join(commom_dir, 'Datasets')
global iter
iter = 1


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
    def __init__(self, file_path, device):
        self.common_path = os.path.join(data_dir, 'NWPU-RESISC45')
        self.file_paths = file_path
        self.patch_size = (128, 128)
        self.scale_extent = (1.25, 2.5)
        self.purtube = 32
        self.erasing_prob = 0.5
        self.border_margin = 16
        self.device = device

    def __getitem__(self, idx):
        global iter
        img_array = io.imread(os.path.join(self.common_path, self.file_paths[idx]))  # h * w *c RGB image array
        # random scale
        scale_factor = rand(self.scale_extent[0], self.scale_extent[1])
        scaled_img = transform.rescale(img_array, scale_factor, multichannel=True)
        # random patch
        h, w = scaled_img.shape[0], scaled_img.shape[1]
        x_limit = (self.purtube + self.border_margin,
                   w - 1 - (self.purtube + self.border_margin) - (self.patch_size[0] - 1) + 1)  # left-top
        corner_x = np.random.randint(x_limit[0], x_limit[1])
        y_limit = (self.purtube + self.border_margin,
                   h - 1 - (self.purtube + self.border_margin) - (self.patch_size[1] - 1) + 1)
        corner_y = np.random.randint(y_limit[0], y_limit[1])
        # random homography
        cornersA = np.float32([[corner_x, corner_y], [corner_x + self.patch_size[0] - 1, corner_y],
                               [corner_x + self.patch_size[0] - 1, corner_y + self.patch_size[1] - 1],
                               [corner_x, corner_y + self.patch_size[1] - 1]])
        patchB = None
        h_ab_4p = rand(-self.purtube, self.purtube, size=(4, 2))
        cornersB = np.float32(cornersA + h_ab_4p)
        h_ab = (transform.estimate_transform('projective', cornersA, cornersB)).params
        h_ab = h_ab / h_ab[2][2]
        h_ba = np.linalg.inv(h_ab)
        patchA = scaled_img[corner_y:corner_y + self.patch_size[1],
                 corner_x:corner_x + self.patch_size[0], :].copy()

        patchB = sk_warpcrop(scaled_img, h_ba, (corner_x, corner_y, self.patch_size[0], self.patch_size[1]))
        '''io.imsave('./experiments/' + str(iter) + 'pA.png', patchA, check_contrast=False)
        io.imsave('./experiments/' + str(iter) + 'pB.png', patchB, check_contrast=False)'''
        # color distortion
        patchA_dst = corlor_distortion(patchA)
        patchA_gray = color.rgb2gray(patchA_dst)
        patchB_gray = color.rgb2gray(patchB)
        # random erasing
        patchA_exp = np.expand_dims(patchA_gray, axis=2)
        patchB_exp = np.expand_dims(patchB_gray, axis=2)
        patchA_re = random_erasing(patchA_exp, self.erasing_prob)
        patchB_re = random_erasing(patchB_exp, self.erasing_prob)
        '''io.imsave('./experiments/' + str(iter) + 'eA.png', patchA_re[..., 0], check_contrast=False)
        io.imsave('./experiments/' + str(iter) + 'eB.png', patchB_re[..., 0], check_contrast=False)'''
        iter += 1
        patchA_tp = patchA_re.transpose((2, 0, 1))
        patchB_tp = patchB_re.transpose((2, 0, 1))
        data_array = np.concatenate([patchA_tp, patchB_tp], axis=0)

        return torch.tensor(data_array, dtype=torch.double, device=self.device), \
               torch.tensor(h_ab_4p, dtype=torch.double, device=self.device).reshape(4 * 2)

    def __len__(self):
        return len(self.file_paths)


def getAerialData():
    pass


def getResiscData(train_proportion=0.8, device='cpu'):
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
    return ResiscDataset(train_files, device), ResiscDataset(val_files, device)
