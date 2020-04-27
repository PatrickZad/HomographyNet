import torch
import cv2
import numpy as np


class CommonCfg:
    def __init__(self, in_height, in_width):
        self.width_in = in_width
        self.height_in = in_height
        self.conv_kernel = 3
        self.padding = 1
        self.in_channel = 2
        self.fc_outs = (1024, 8)
        self.conv_channels = None
        self.fc_in = None
        self.dropout_prob = (0.5, 0.5)
        self.batch_size = 64
        self.base_lr = 0.005
        self.lr_decrease = {'factor': 0.1, 'interval': 30000}
        self.momentum = 0.9
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_epoch = 256
        self.save_period = 10
        self.log_period=50

    def get_optimizer(self, params):
        return torch.optim.SGD(params, lr=self.base_lr, momentum=self.momentum)

    def get_lr_scheduler(self, optimizer, last_epoch=-1):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_decrease['interval'],
                                               gamma=self.lr_decrease['factor'], last_epoch=last_epoch)

    def get_loss(self):
        return torch.nn.MSELoss()


class Cfg128(CommonCfg):
    def __init__(self):
        super(Cfg128, self).__init__(128, 128)
        self.conv_out_channels = [self.in_channel] + [64, 64, 64, 64, 128, 128, 128, 128]
        self.fc_in = (self.height_in // 8) * (self.width_in // 8) * self.conv_out_channels[-1]


class Cfg256(CommonCfg):
    def __init__(self):
        super(Cfg256).__init__(256, 256)
        self.conv_channels = [self.in_channel] + [64, 64, 128, 128, 128, 128, 256, 256, 256, 256]
        self.fc_in = (self.height_in // 16) * (self.width_inh // 16) * self.conv_out_channels[-1]


def rand(a=0, b=1, size=None):
    return np.random.rand(*size) * (b - a) + a


def corlor_distortion(rgb_array, hue=.1, sat=1.5, val=1.5):
    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = cv2.cvtColor(rgb_array / 255., cv2.COLOR_RGB2HSV)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    return cv2.cvtColor(x, cv2.COLOR_HSV2RGB)


def warpcrop_in_same_coordsys(img, homo_mat, warpcrop_box):
    corners = np.float32([[0, 0], [img.shape[1] - 1, 0], [
        0, img.shape[0] - 1], [img.shape[1] - 1, img.shape[0] - 1]])
    warpped_corners = np.matmul(corners, homo_mat.T)
    x, y, w, h = cv2.boundingRect(warpped_corners)
    translation_corner = (warpcrop_box[1] - x, warpcrop_box[0] - y)
    warpped_img = cv2.warpPerspective(img, homo_mat, (w, h))
    return warpped_img[translation_corner[1]:translation_corner[1] + warpcrop_box[1],
           translation_corner[0]:translation_corner[0] + warpcrop_box[0], :]


def random_erasing(img, prob=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465), max_iter=100):
    if np.random.uniform(0, 1) >= prob:
        return img
    for i in range(max_iter):
        area = img.shape[0] * img.shape[1]

        region_area = np.random.uniform(sl, sh) * area
        aspect_ratio = np.random.uniform(r1, 1 / r1)
        h = int(round((region_area * aspect_ratio) ** 0.5))
        w = int(round((region_area / aspect_ratio) ** 0.5))

        if w < img.shape[1] and h < img.shape[0]:
            x1 = np.random.randint(0, img.size()[1] - h)
            y1 = np.random.randint(0, img.size()[2] - w)
            if img.size()[0] == 3:
                img[0, x1:x1 + h, y1:y1 + w] = mean[0]
                img[1, x1:x1 + h, y1:y1 + w] = mean[1]
                img[2, x1:x1 + h, y1:y1 + w] = mean[2]
            else:
                img[0, x1:x1 + h, y1:y1 + w] = mean[0]
            return img
    return img
