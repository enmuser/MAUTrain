import torch
import torch.nn as nn
from core.layers.MAUCell import MAUCell
import math


class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()
        self.configs = configs
        # patch_size = 1
        # frame_channel = 1 * 1 * 1 = 1
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        # num_layers = 4
        self.num_layers = num_layers
        # num_hidden = [64,64,64,64]
        self.num_hidden = num_hidden
        # tau = 5
        self.tau = configs.tau
        self.cell_mode = configs.cell_mode
        self.states = ['recall', 'normal']
        if not self.configs.model_mode in self.states:
            raise AssertionError
        cell_list = []
        # sr_size = 4
        # width = 64 / 1 / 4 = 16
        # height = 64 / 1 / 4 = 16
        width = configs.img_width // configs.patch_size // configs.sr_size
        height = configs.img_height // configs.patch_size // configs.sr_size

        for i in range(num_layers):
            in_channel = num_hidden[i - 1]
            # in_channel = 1 ,num_hidden[i] = 64,  height = 16 , width = 16,
            # filter_size = (5,5), stride = 1,tau = 5,cell_mode = normal
            cell_list.append(
                MAUCell(in_channel, num_hidden[i], height, width, configs.filter_size,
                        configs.stride, self.tau, self.cell_mode)
            )
        self.cell_list = nn.ModuleList(cell_list)

        # Encoder
        # math.log2(4) = 2
        n = int(math.log2(configs.sr_size))
        encoders = []
        encoder = nn.Sequential()
        encoder.add_module(name='encoder_t_conv{0}'.format(-1),
                           # frame_channel = 1, num_hidden = 64
                           module=nn.Conv2d(in_channels=self.frame_channel,
                                            out_channels=self.num_hidden[0],
                                            stride=1,
                                            padding=0,
                                            kernel_size=1))
        encoder.add_module(name='relu_t_{0}'.format(-1),
                           module=nn.LeakyReLU(0.2))
        encoders.append(encoder)
        for i in range(n):
            # 每次图片大小减掉一倍
            encoder = nn.Sequential()
            encoder.add_module(name='encoder_t{0}'.format(i),
                               # in_channels = 64, out_channels = 64, stride = (2,2),padding = (1,1), kernel_size= (3,3)
                               # outshape = 每次图片大小减掉一半
                               module=nn.Conv2d(in_channels=self.num_hidden[0],
                                                out_channels=self.num_hidden[0],
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                kernel_size=(3, 3)
                                                ))
            encoder.add_module(name='encoder_t_relu{0}'.format(i),
                               module=nn.LeakyReLU(0.2))
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        # Decoder
        decoders = []

        # n = 2
        for i in range(n - 1):
            # 每次图片大小增加一倍
            # H_in = 16
            # W_in = 16
            # stride = (2, 2)
            # padding = (1, 1)
            # kernel_size = (3, 3)
            # output_padding = (1, 1)
            # dilation 默认 = 1
            # 基于转置卷积的计算公式
            # H_out = (16 - 1) * 2 - 2 * 1 + 1 * (3 - 1) + 1 + 1 = 32 - 2 - 2 + 2 + 2 = 32
            # W_out = (16 - 1) * 2 - 2 * 1 + 1 * (3 - 1) + 1 + 1 = 32 - 2 - 2 + 2 + 2 = 32
            # 大小扩充一倍的上采样
            decoder = nn.Sequential()
            decoder.add_module(name='c_decoder{0}'.format(i),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(1, 1)
                                                         ))
            decoder.add_module(name='c_decoder_relu{0}'.format(i),
                               module=nn.LeakyReLU(0.2))
            decoders.append(decoder)

        if n > 0:
            decoder = nn.Sequential()
            decoder.add_module(name='c_decoder{0}'.format(n - 1),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(1, 1)
                                                         ))
            decoders.append(decoder)
        self.decoders = nn.ModuleList(decoders)

        # channel => 64 -> 1
        self.srcnn = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1], self.frame_channel, kernel_size=1, stride=1, padding=0)
        )
        # channel => 64 * 2 -> 64
        self.merge = nn.Conv2d(self.num_hidden[-1] * 2, self.num_hidden[-1], kernel_size=1, stride=1, padding=0)
        # channel => 2 -> 1
        self.conv_last_sr = nn.Conv2d(self.frame_channel * 2, self.frame_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, frames, mask_true):
        # print('ok')
        # 1. frames 图片信息 16 * 20 * 1 * 64 * 64
        # 2. mask_true real_input_flag掩码信息 16 * 9 * 64 * 64 * 1
        # mask_true => 16 * 9 * 64 * 64 * 1 -> 16 * 9 * 1 * 64 * 64
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        # batch_size = 16
        batch_size = frames.shape[0]
        # height = 64 / 4 = 16
        # width = 64 / 4 = 16
        height = frames.shape[3] // self.configs.sr_size
        width = frames.shape[4] // self.configs.sr_size
        frame_channels = frames.shape[2]
        next_frames = []
        T_t = []
        T_pre = []
        S_pre = []
        x_gen = None
        # num_layers = 0, 1, 2, 3
        for layer_idx in range(self.num_layers):
            tmp_t = []
            tmp_s = []
            if layer_idx == 0:
                # in_channel = 64
                in_channel = self.num_hidden[layer_idx]
            else:
                # in_channel = 64
                in_channel = self.num_hidden[layer_idx - 1]
            # tau= 5 : 0, 1, 2, 3, 4
            for i in range(self.tau):
                tmp_t.append(torch.zeros([batch_size, in_channel, height, width]).to(self.configs.device))# 16 * 64 * 16 * 16
                tmp_s.append(torch.zeros([batch_size, in_channel, height, width]).to(self.configs.device))# 16 * 64 * 16 * 16
            T_pre.append(tmp_t) # 4 * 5 * 16 * 64 * 16 * 16
            S_pre.append(tmp_s) # 4 * 5 * 16 * 64 * 16 * 16
        # total_length = 20,  0,1,2,3,......,16,17,18
        for t in range(self.configs.total_length - 1):
            # input_length = 10
            if t < self.configs.input_length:
                # frames[:, t] = 16 * 1 * 1 * 64 * 64 = 16 * 1 * 64 * 64
                net = frames[:, t]
            else:
                # example: t = 10, input_length = 10
                # time_diff = 0
                time_diff = t - self.configs.input_length
                # mask_true[:, time_diff] = 16 * 1 * 64 * 64
                # frames[:, t] = 16 * 1 * 64 * 64
                # mask_true[:, time_diff] * frames[:, t] = 16 * 1 * 64 * 64
                # 是个原始图片数据 或者 全是0  最开始大概率是 0
                # 相对应的 (1 - mask_true[:, time_diff]) 大概率为 1
                # x_gen = 16 * 1 * 64 * 64 预测的下一帧
                # (1 - mask_true[:, time_diff]) * x_gen = 16 * 1 * 64 * 64
                net = mask_true[:, time_diff] * frames[:, t] + (1 - mask_true[:, time_diff]) * x_gen
            # frames_feature = 16 * 1 * 64 * 64
            frames_feature = net
            frames_feature_encoded = []
            for i in range(len(self.encoders)):
                # 1. 16 * 1 * 64 * 64 -> 16 * 64 * 64 * 64 => frames_feature_encoded
                # 2. 16 * 64 * 64 * 64 -> 16 * 64 * 32 * 32 => frames_feature_encoded
                # 3. 16 * 64 * 32 * 32 -> 16 * 64 * 16 * 16 => frames_feature_encoded
                frames_feature = self.encoders[i](frames_feature)
                frames_feature_encoded.append(frames_feature)
            if t == 0:
                # num_layers = 4
                # 0, 1, 2, 3
                for i in range(self.num_layers):
                    zeros = torch.zeros([batch_size, self.num_hidden[i], height, width]).to(self.configs.device) # 16 * 64 * 16 * 16
                    T_t.append(zeros)# 4 * 16 * 64 * 16 * 16
            S_t = frames_feature # 16 * 64 * 16 * 16
            # num_layers = 4
            # 0, 1, 2, 3
            for i in range(self.num_layers):
                t_att = T_pre[i][-self.tau:] #
                t_att = torch.stack(t_att, dim=0) # 5 * 16 * 64 * 16 * 16
                s_att = S_pre[i][-self.tau:]
                s_att = torch.stack(s_att, dim=0) # 5 * 16 * 64 * 16 * 16
                S_pre[i].append(S_t)
                T_t[i], S_t = self.cell_list[i](T_t[i], S_t, t_att, s_att)
                T_pre[i].append(T_t[i])
            out = S_t
            # out = self.merge(torch.cat([T_t[-1], S_t], dim=1))
            frames_feature_decoded = []
            for i in range(len(self.decoders)):
                # 1. 16 * 64 * 16 * 16 -> 16 * 64 * 32 * 32
                # 2. 16 * 64 * 32 * 32 -> 16 * 64 * 64 * 64
                out = self.decoders[i](out)
                if self.configs.model_mode == 'recall':
                    out = out + frames_feature_encoded[-2 - i]

            x_gen = self.srcnn(out) # 16 * 64 * 64 * 64 => # 16 * 1 * 64 * 64
            next_frames.append(x_gen)
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        return next_frames # 16 * 19 * 1 * 64 * 64
