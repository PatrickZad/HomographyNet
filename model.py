import torch.nn as nn
import torch


def net_block(in_channel, out_channel, index):
    if index % 2 == 0:
        return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
                             nn.BatchNorm2d(out_channel),
                             nn.ReLU())
    else:
        return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
                             nn.BatchNorm2d(out_channel),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2))


class HomographyNet(nn.Module):
    def __init__(self, cfg):
        super(HomographyNet, self).__init__()
        self.cfg = cfg
        conv_nums = len(cfg.conv_out_channels)
        self.conv_component = nn.Sequential(*[net_block(cfg.conv_out_channels[i],
                                                        cfg.conv_out_channels[i + 1], i)
                                              for i in range(conv_nums - 1)])
        self.conv_component = nn.Sequential(self.conv_component, nn.Dropout2d(cfg.dropout_prob[0]))
        self.fc1_2 = nn.Sequential(nn.Linear(cfg.fc_in, cfg.fc_outs[0]),
                                   nn.Dropout2d(cfg.dropout_prob[1]),
                                   nn.Linear(cfg.fc_outs[0], cfg.fc_outs[1]))

    def forward(self, img_pair):
        conv_out = self.conv_component(img_pair)
        feature1d = torch.reshape(conv_out, (-1,))
        return self.fc1_2(feature1d)
