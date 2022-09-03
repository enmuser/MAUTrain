import torch
from torch import nn

from core.models.Inception import Inception


class Inception_decoder(nn.Module):
    def __init__(self, shape_in, hid_S=16, hid_T=256, N_S=4, N_T=8, incep_ker=[3,5,7,11], groups=8):
        # channel_in = T * hid_S = 10 * 64 = 640,
        # channel_hid = hid_T = 256,
        # N_T = N_T = 8,
        # incep_ker = [3, 5, 7, 11],
        # groups = 8
        super(Inception_decoder, self).__init__()

        T, C, H, W = shape_in  # [T = 10, C = 1, H = 64, W = 64]
        channel_in = T * hid_S
        channel_hid = hid_T

        # N_T = 8
        self.N_T = N_T
        # channel_hid = 256, channel_hid // 2 = 128, channel_hid = 256, incep_ker=[3,5,7,11], groups = 8
        # deconder_layers_0=(channel_hid = 256, channel_hid // 2 = 128, channel_hid = 256, incep_ker=[3,5,7,11], groups = 8)
        dec_layers = [Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            # deconder_layers_1_6=(channel_hid = 256, channel_hid // 2 = 128, channel_hid = 256, incep_ker=[3,5,7,11], groups = 8)
            dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        # deconder_layers_7 =(channel_hid = 256, channel_hid // 2 = 128, channel_hid = 640, incep_ker=[3,5,7,11], groups = 8)
        dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_in, incep_ker= incep_ker, groups=groups))
        # encoder:
        #
        # Sequential(
        #     (0): GroupConv2d(
        #     (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=8)
        # (norm): GroupNorm(8, 256, eps=1e-05, affine=True)
        # (activate): LeakyReLU(negative_slope=0.2, inplace=True)
        # )
        # (1): GroupConv2d(
        #     (conv): Conv2d(128, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=8)
        # (norm): GroupNorm(8, 256, eps=1e-05, affine=True)
        # (activate): LeakyReLU(negative_slope=0.2, inplace=True)
        # )
        # (2): GroupConv2d(
        #     (conv): Conv2d(128, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=8)
        # (norm): GroupNorm(8, 256, eps=1e-05, affine=True)
        # (activate): LeakyReLU(negative_slope=0.2, inplace=True)
        # )
        # (3): GroupConv2d(
        #     (conv): Conv2d(128, 256, kernel_size=(11, 11), stride=(1, 1), padding=(5, 5), groups=8)
        # (norm): GroupNorm(8, 256, eps=1e-05, affine=True)
        # (activate): LeakyReLU(negative_slope=0.2, inplace=True)
        # )
        # )
        #
        #
        # decoder_0
        #
        # Sequential(
        #     (0): GroupConv2d(
        #     (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=8)
        # (norm): GroupNorm(8, 256, eps=1e-05, affine=True)
        # (activate): LeakyReLU(negative_slope=0.2, inplace=True)
        # )
        # (1): GroupConv2d(
        #     (conv): Conv2d(128, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=8)
        # (norm): GroupNorm(8, 256, eps=1e-05, affine=True)
        # (activate): LeakyReLU(negative_slope=0.2, inplace=True)
        # )
        # (2): GroupConv2d(
        #     (conv): Conv2d(128, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=8)
        # (norm): GroupNorm(8, 256, eps=1e-05, affine=True)
        # (activate): LeakyReLU(negative_slope=0.2, inplace=True)
        # )
        # (3): GroupConv2d(
        #     (conv): Conv2d(128, 256, kernel_size=(11, 11), stride=(1, 1), padding=(5, 5), groups=8)
        # (norm): GroupNorm(8, 256, eps=1e-05, affine=True)
        # (activate): LeakyReLU(negative_slope=0.2, inplace=True)
        # )
        # )
        #
        # decoder_1_6
        #
        # Sequential(
        #     (0): GroupConv2d(
        #     (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=8)
        # (norm): GroupNorm(8, 256, eps=1e-05, affine=True)
        # (activate): LeakyReLU(negative_slope=0.2, inplace=True)
        # )
        # (1): GroupConv2d(
        #     (conv): Conv2d(128, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=8)
        # (norm): GroupNorm(8, 256, eps=1e-05, affine=True)
        # (activate): LeakyReLU(negative_slope=0.2, inplace=True)
        # )
        # (2): GroupConv2d(
        #     (conv): Conv2d(128, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=8)
        # (norm): GroupNorm(8, 256, eps=1e-05, affine=True)
        # (activate): LeakyReLU(negative_slope=0.2, inplace=True)
        # )
        # (3): GroupConv2d(
        #     (conv): Conv2d(128, 256, kernel_size=(11, 11), stride=(1, 1), padding=(5, 5), groups=8)
        # (norm): GroupNorm(8, 256, eps=1e-05, affine=True)
        # (activate): LeakyReLU(negative_slope=0.2, inplace=True)
        # )
        # )
        #
        # decoder_7
        # Sequential(
        #     (0): GroupConv2d(
        #     (conv): Conv2d(128, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=8)
        # (norm): GroupNorm(8, 640, eps=1e-05, affine=True)
        # (activate): LeakyReLU(negative_slope=0.2, inplace=True)
        # )
        # (1): GroupConv2d(
        #     (conv): Conv2d(128, 640, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=8)
        # (norm): GroupNorm(8, 640, eps=1e-05, affine=True)
        # (activate): LeakyReLU(negative_slope=0.2, inplace=True)
        # )
        # (2): GroupConv2d(
        #     (conv): Conv2d(128, 640, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=8)
        # (norm): GroupNorm(8, 640, eps=1e-05, affine=True)
        # (activate): LeakyReLU(negative_slope=0.2, inplace=True)
        # )
        # (3): GroupConv2d(
        #     (conv): Conv2d(128, 640, kernel_size=(11, 11), stride=(1, 1), padding=(5, 5), groups=8)
        # (norm): GroupNorm(8, 640, eps=1e-05, affine=True)
        # (activate): LeakyReLU(negative_slope=0.2, inplace=True)
        # )
        # )
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x, skips):
        # B, T, C, H, W = x.shape  # 16 * 10 * 64 * 16 * 16 ===> B = 16, T = 10, C = 64, H = 16, W = 16
        # #  B = 16, T = 10, C = 64, H = 16, W = 16
        # x = x.reshape(B, T*C, H, W)
        # x = 16 * 10 * 64 * 16 * 16 ==> x = 16 * 640 * 16 * 16
        z = x # z = 16 * 640 * 16 * 16


        # decoder
        # z = 16 * 256 * 16 * 16
        z = self.dec[0](z)
        # z = 16 * 256 * 16 * 16
        for i in range(1, self.N_T):
            # 输入: torch.cat([z, skips[-i]], dim=1) = 16 * 512 * 16 * 16
            # 输出:
            # 前6次:z = 16 * 256 * 16 * 16
            # 最后第7次:z = 16 * 640 * 16 * 16
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))
        # z = 16 * 640 * 16 * 16
        #  B = 16, T = 10, C = 64, H = 16, W = 16
        # y = 16 * 10 * 64 * 16 * 16
        # y = z.reshape(B, T, C, H, W)
        return z
