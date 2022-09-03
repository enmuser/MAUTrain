from torch import nn


class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, act_norm=False):
        super(GroupConv2d, self).__init__()
        # GroupConv2d_3:
        # 入参:  C_hid = 128, C_out = 128, kernel_size=3, stride=1, padding=3//2, groups=8, act_norm=True
        # ----------------------------------------------------------------------------------------------------------------------
        # in_channels = C_hid = 128, out_channels = C_out = 256, kernel_size = 3, stride = 1, padding = 1, groups = 8, act_norm=True
        # in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, groups = 8, act_norm=True
        # conv: (16-3+2*1)/1 + 1 = 15 / 1 + 1 = 16
        # 作用 16 * 128 * 16 * 16 ——> 16 * 256 * 16 * 16
        # GroupConv2d_5:
        # 入参:  C_hid = 128, C_out = 256, kernel_size=5, stride=1, padding=5//2, groups=8, act_norm=True
        # in_channels = 128, out_channels = 128, kernel_size = 5, stride = 1, padding = 2, groups = 8, act_norm = True
        # conv: (16-5+2*2)/1 + 1 = 15 / 1 + 1 = 16
        # 作用 16 * 128 * 16 * 16 ——> 16 * 256 * 16 * 16
        # GroupConv2d_7:
        # 入参:  C_hid = 128, C_out = 128, kernel_size=7, stride=1, padding=7//2, groups=8, act_norm=True
        # in_channels = 128, out_channels = 256, kernel_size = 7, stride = 1, padding = 3, groups = 8, act_norm = True
        # conv: (16-7+3*2)/1 + 1= 15 / 1 + 1 = 16
        # 作用 16 * 128 * 16 * 16 ——> 16 * 256 * 16 * 16
        # GroupConv2d_11:
        # 入参:  C_hid = 128, C_out = 128, kernel_size=11, stride=1, padding=11//2, groups=8, act_norm=True
        # in_channels = 128, out_channels = 256, kernel_size = 11, stride = 1, padding = 5, groups = 8, act_norm = True
        # conv: (16-11+2 * 5)/1 + 1 = 15 / 1 + 1= 16
        # 作用 16 * 128 * 16 * 16 ——> 16 * 256 * 16 * 16
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=groups)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y