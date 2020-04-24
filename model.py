import torch.nn as nn
import torch


def net_block(in_channel, out_channel):
    return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
                         nn.BatchNorm2d(out_channel),
                         nn.ReLU())


class HomographyEstimator(nn.Module):
    def __init__(self):
        super(HomographyEstimator, self).__init__()
        self.conv1_2 = nn.Sequential(net_block(2, 64), net_block(64, 64), nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv3_4 = nn.Sequential(net_block(64, 64), net_block(64, 64), nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv5_6 = nn.Sequential(net_block(64, 128), net_block(128, 128), nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv7_8 = nn.Sequential(net_block(128, 128), net_block(128, 128), nn.Dropout2d(0.5))
        self.fc1_2 = nn.Sequential(nn.Linear(16 * 16 * 128, 1024), nn.Dropout2d(0.5), nn.Linear(1024, 8))

    def forward(self, img_pair):
        out1 = self.conv1_2(img_pair)
        out2 = self.conv3_4(out1)
        out3 = self.conv5_6(out2)
        out4 = self.conv7_8(out3)
        feature1d = torch.reshape(out4, (-1,))
        return self.fc1_2(feature1d)
