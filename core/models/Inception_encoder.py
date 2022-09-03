import torch
from torch import nn

from core.models.Inception import Inception


class Inception_encoder(nn.Module):

    def __init__(self, shape_in, hid_S=16, hid_T=256, N_S=4, N_T=8, incep_ker=[3, 5, 7, 11], groups=8):
        super(Inception_encoder, self).__init__()

        # shape_in = in_shape=[10,1,64,64]
        # hid_S = 64
        # hid_T = 256
        # N_S = 4
        # N_T = 8
        T, C, H, W = shape_in  # [T = 10, C = 1, H = 64, W = 64]
        channel_in = T * hid_S
        channel_hid = hid_T
        # N_T = 8
        self.N_T = N_T
        # enconder_layers_0=(channel_in = 640, channel_hid//2 = 128, channel_hid = 256, incep_ker=[3,5,7,11], groups=8)
        enc_layers = [Inception(channel_in, channel_hid // 2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T - 1):
            # enconder_layers_1_6=(channel_in = 256, channel_hid//2 = 128, channel_hid = 256, incep_ker=[3,5,7,11], groups=8)
            enc_layers.append(Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker= incep_ker, groups=groups))
        # enconder_layers_7=(channel_in = 256, channel_hid//2 = 128, channel_hid = 256, incep_ker=[3,5,7,11], groups=8)
        enc_layers.append(Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker= incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape  # 16 * 10 * 64 * 16 * 16 ===> B = 16, T = 10, C = 64, H = 16, W = 16
        #  B = 16, T = 10, C = 64, H = 16, W = 16
        x = x.reshape(B, T * C, H, W)
        # x = 16 * 10 * 64 * 16 * 16 ==> x = 16 * 640 * 16 * 16

        # encoder
        skips = []
        z = x  # z = 16 * 640 * 16 * 16
        for i in range(self.N_T):
            # 第一次输入是: z = 16 * 640 * 16 * 16
            # 后面都是: z = 16 * 256 * 16 * 16
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)

        return z, skips
