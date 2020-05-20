import torch
import cv2
import numpy as np
from skimage import color, util, io, transform


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
        self.batch_size = 128
        self.base_lr = 0.0001
        self.lr_decrease = {'factor': 0.1, 'interval': 10}
        self.momentum = 0.9
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_epoch = 200
        self.save_period = 5
        self.log_period = 10
        self.multi_gpu = False

    def get_optimizer(self, params):
        return torch.optim.SGD(params, lr=self.base_lr, momentum=self.momentum)

    def get_lr_scheduler(self, optimizer, last_epoch=-1):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                               factor=self.lr_decrease['factor'], patience=5)

    def get_loss(self):
        return torch.nn.MSELoss()


class Cfg128(CommonCfg):
    def __init__(self):
        super(Cfg128, self).__init__(128, 128)
        self.conv_channels = [self.in_channel] + [64, 64, 64, 64, 128, 128, 128, 128]
        self.fc_in = (self.height_in // 8) * (self.width_in // 8) * self.conv_channels[-1]


class Cfg256(CommonCfg):
    def __init__(self):
        super(Cfg256).__init__(256, 256)
        self.conv_channels = [self.in_channel] + [64, 64, 128, 128, 128, 128, 256, 256, 256, 256]
        self.fc_in = (self.height_in // 16) * (self.width_inh // 16) * self.conv_out_channels[-1]


def rand(a=0, b=1, size=None):
    if size is not None:
        return np.random.rand(*size) * (b - a) + a
    else:
        return np.random.rand() * (b - a) + a


def corlor_distortion(rgb_array, hue=.1, sat=1.5, val=1.5):
    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    # cv_float_img = np.float32(rgb_array) / 255.
    float_img = util.img_as_float(rgb_array)
    # x = cv2.cvtColor(float_img, cv2.COLOR_RGB2HSV)
    x = color.rgb2hsv(float_img)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    # cv_result = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)
    result = color.hsv2rgb(x)
    return result


global count
count = 1


def sk_warpcrop(img, homo_mat, warpcrop_box):
    global count
    warpped_img = transform.warp(img, homo_mat)
    crop = warpped_img[warpcrop_box[1]:warpcrop_box[1] + warpcrop_box[3],
           warpcrop_box[0]:warpcrop_box[0] + warpcrop_box[2], :].copy()
    crop_h, crop_w = crop.shape[0], crop.shape[1]
    if crop_h != warpcrop_box[3] or crop_w != warpcrop_box[2]:
        print('Regenerate random patch !')
        return None
    '''io.imsave('./experiments/' + str(count) + 'A.png', img, check_contrast=False)
    io.imsave('./experiments/' + str(count) + 'B.png', warpped_img, check_contrast=False)'''
    count += 1
    return crop


def warpcrop_in_same_coordsys(img, homo_mat, warpcrop_box, polyA, polyB):
    global count

    corners = np.float32([[0, 0, 1], [img.shape[1] - 1, 0, 1],
                          [img.shape[1] - 1, img.shape[0] - 1, 1],
                          [0, img.shape[0] - 1, 1]])
    scaled_homo_mat = homo_mat / homo_mat[2][2]
    warpped_corners = np.matmul(corners, scaled_homo_mat.T)
    warpped_corners /= warpped_corners[:, 2:]
    warpped_corners = np.int32(warpped_corners[:, :-1])
    x, y, w, h = cv2.boundingRect(warpped_corners)
    translation_corner = (warpcrop_box[0] - x, warpcrop_box[1] - y)
    translation_mat = np.eye(3)
    translation_mat[0][2], translation_mat[1][2] = -x, -y
    warp_mat = np.matmul(translation_mat, scaled_homo_mat)
    trans_warp_corners = np.matmul(corners, warp_mat.T)
    trans_warp_corners /= trans_warp_corners[:, 2:]
    try:
        warpped_img = cv2.warpPerspective(img, warp_mat, (w, h))
    except BaseException as e:
        # print(e)
        return None
    crop = warpped_img[translation_corner[1]:translation_corner[1] + warpcrop_box[3],
           translation_corner[0]:translation_corner[0] + warpcrop_box[2], :].copy()
    crop_h, crop_w = crop.shape[0], crop.shape[1]
    warpB = np.matmul(np.concatenate([polyB, np.ones((4, 1))], axis=-1), warp_mat.T)
    warpB /= warpB[:, 2:]
    if crop_h != warpcrop_box[3] or crop_w != warpcrop_box[2]:
        # print('Regenerate random patch !')
        return None
    '''img = cv2.polylines(img, polyB.reshape((-1, 1, 2)), isClosed=True, color=(0, 255, 0), lineType=cv2.LINE_8,
                        thickness=2)
    img = cv2.polylines(img, polyA.reshape((-1, 1, 2)), isClosed=True, color=(0, 0, 255), lineType=cv2.LINE_8,
                        thickness=2)
    cv2.imwrite('./experiments/' + str(count) + 'A.png', img)
    cv2.imwrite('./experiments/' + str(count) + 'B.png', warpped_img)
    '''

    count += 1
    return crop


def random_erasing(img, prob=0.5, sl=0.02, sh=0.15, r1=0.3, mean=(0.4914, 0.4822, 0.4465), max_iter=100):
    if np.random.uniform(0, 1) >= prob:
        return img
    for i in range(max_iter):
        area = img.shape[0] * img.shape[1]

        region_area = np.random.uniform(sl, sh) * area
        aspect_ratio = np.random.uniform(r1, 1 / r1)
        h = int(round((region_area * aspect_ratio) ** 0.5))
        w = int(round((region_area / aspect_ratio) ** 0.5))

        if w < img.shape[1] and h < img.shape[0]:
            y1 = np.random.randint(0, img.shape[0] - h)
            x1 = np.random.randint(0, img.shape[1] - w)
            if img.shape[-1] == 3:
                img[y1:y1 + h, x1: x1 + w, :] = np.array(mean)
            else:
                img[y1:y1 + h, x1:x1 + w, :] = np.array([mean[0]])
            return img
    return img
