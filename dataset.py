import os
import torch
from torch.utils.data import Dataset
import cv2
from common import *

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
    def __init__(self, file_path):
        self.common_path = os.path.join(data_dir, 'NWPU-RESISC45')
        self.file_paths = file_path
        self.patch_size = (128, 128)
        self.scale_extent = (1.25, 2.5)
        self.purtube = 32
        self.erasing_prob = 0.5
        self.border_margin = 16

    def __getitem__(self, idx):
        global iter
        img_array = cv2.imread(os.path.join(self.common_path, self.file_paths[idx]))

        # random scale
        scale_factor = rand(self.scale_extent[0], self.scale_extent[1])
        scaled_img = cv2.resize(img_array, (0, 0), fx=scale_factor, fy=scale_factor)
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
        h_ab_4p = rand(-self.purtube, self.purtube, size=(4, 2))
        cornersB = np.float32(cornersA + h_ab_4p)
        '''cornersA = cornersA.reshape((-1, 1, 2))
        cornersB = cornersB.reshape((-1, 1, 2))'''
        h_ab = cv2.getPerspectiveTransform(cornersA, cornersB)
        h_ab = h_ab / h_ab[2][2]
        h_ba = np.linalg.inv(h_ab)
        # h_ba = h_ba / h_ba[2][2]
        patchA = scaled_img[corner_y:corner_y + self.patch_size[1], corner_x:corner_x + self.patch_size[0], :].copy()
        patchB = warpcrop_in_same_coordsys(scaled_img, h_ba,
                                           (corner_x, corner_y, self.patch_size[0], self.patch_size[1]),
                                           polyA=np.int32(cornersA),
                                           polyB=np.int32(cornersB))
        while patchB is None:
            h_ab_4p = rand(-self.purtube, self.purtube, size=(4, 2))
            cornersB = np.float32(cornersA + h_ab_4p)
            h_ab = cv2.getPerspectiveTransform(cornersA, cornersB)
            h_ab = h_ab / h_ab[2][2]
            h_ba = np.linalg.inv(h_ab)
            # h_ba = h_ba / h_ba[2][2]
            patchA = scaled_img[corner_y:corner_y + self.patch_size[1], corner_x:corner_x + self.patch_size[0],
                     :].copy()
            patchB = warpcrop_in_same_coordsys(scaled_img, h_ba,
                                               (corner_x, corner_y, self.patch_size[0], self.patch_size[1]),
                                               polyA=np.int32(cornersA),
                                               polyB=np.int32(cornersB))
        '''cv2.imwrite('./experiments/' + str(iter) + 'A.png', patchA)

        cv2.imwrite('./experiments/' + str(iter) + 'B.png', patchB)'''
        iter += 1
        # color distortion
        patchA = cv2.cvtColor(patchA, cv2.COLOR_BGR2RGB)
        patchA_dst = corlor_distortion(patchA)
        patchA_gray = cv2.cvtColor(patchA_dst, cv2.COLOR_RGB2GRAY)
        patchB_gray = np.float32(cv2.cvtColor(patchB, cv2.COLOR_BGR2GRAY)) / 255

        # random erasing
        patchA_exp = np.expand_dims(patchA_gray, axis=2)
        patchB_exp = np.expand_dims(patchB_gray, axis=2)
        patchA_re = random_erasing(patchA_exp, self.erasing_prob)
        patchB_re = random_erasing(patchB_exp, self.erasing_prob)
        patchA_tp = patchA_re.transpose((2, 0, 1))
        patchB_tp = patchB_re.transpose((2, 0, 1))
        data_array = np.concatenate([patchA_tp, patchB_tp], axis=0)
        return torch.tensor(data_array, dtype=torch.double), \
               torch.tensor(h_ab_4p, dtype=torch.double).reshape(4 * 2)

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
