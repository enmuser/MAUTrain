from torch import nn

from core.models.GroupConv2d import GroupConv2d


class Inception(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker = [3,5,7,11], groups=8):
        super(Inception, self).__init__()
        # enconder_layers_0=(channel_in = 640, channel_hid//2 = 128, channel_hid = 256, incep_ker=[3,5,7,11], groups=8)
        # C_in=channel_in = 640, C_hid=channel_hid//2 = 128, C_out=channel_hid = 256, incep_ker=[3,5,7,11], groups=8
        # C_in=640, C_hid=128, C_out=256, incep_ker=[3,5,7,11], groups=8
        #------------------------------------------Encoder-0-conv1--------------------------------------------------------
        # conv1:
        # 入参:  C_in = 640, C_hid = 128 , kernel_size=1, stride=1, padding=0
        # 1. 16 * 640 * 16 * 16 --> 16 * 128 * 16 * 16
        # GroupConv2d_3:
        # 入参:  C_hid = 128, C_out = 256, kernel_size=3, stride=1, padding=3//2, groups=8, act_norm=True
        # GroupConv2d_5:
        # 入参:  C_hid = 128, C_out = 256, kernel_size=3, stride=1, padding=3//2, groups=8, act_norm=True
        # GroupConv2d_7:
        # 入参:  C_hid = 128, C_out = 256, kernel_size=3, stride=1, padding=3//2, groups=8, act_norm=True
        # GroupConv2d_11:
        # 入参:  C_hid = 128, C_out = 256, kernel_size=3, stride=1, padding=3//2, groups=8, act_norm=True

        # enconder_layers_1_6=(channel_in = 256, channel_hid//2 = 128, channel_hid = 256, incep_ker=[3,5,7,11], groups=8)
        # C_in=256, C_hid=128, C_out=256, incep_ker=[3,5,7,11], groups=8

        # enconder_layers_7=(channel_in = 256, channel_hid//2 = 128, channel_hid = 256, incep_ker=[3,5,7,11], groups=8)
        # C_in=256, C_hid=128, C_out=256, incep_ker=[3,5,7,11], groups=8

        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        layers = []
        # incep_ker=[3,5,7,11]
        for ker in incep_ker:
            layers.append(GroupConv2d(C_hid, C_out, kernel_size=ker, stride=1, padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # -------------------encoder--------------------------------
        # 第一次:
        # x = 16 * 640 * 16 * 16 --> 16 * 128 * 16 * 16
        # x = 16 * 128 * 16 * 16
        # 后面:
        # x = 16 * 256 * 16 * 16 --> 16 * 128 * 16 * 16
        # x = 16 * 128 * 16 * 16
        # -------------------decoder--------------------------------
        # 第一次 :
        # x = 16 * 256 * 16 * 16 --> 16 * 128 * 16 * 16
        # x = 16 * 128 * 16 * 16
        # 后面
        # x = 16 * 512 * 16 * 16 --> 16 * 128 * 16 * 16
        # x = 16 * 128 * 16 * 16
        x = self.conv1(x)
        y = 0
        # incep_ker=[3,5,7,11] 四个不同的 kernel 累加结果y
        for layer in self.layers:
            # x = 16 * 256 * 16 * 16 <--- 16 * 128 * 16 * 16
            y += layer(x)
        return y