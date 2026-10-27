import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn.functional import conv1d
import pywt
import torch.nn.functional as F
import numpy as np
from pytorch_wavelets import DWTForward
from kan import KAN
# from .lstm_warehouse import LSTM_Bi
# from .lstm_warehouse import CNN_BiGRU_Attention
# from .Mamba_v3 import MSBlock,FSSBlock,Spec_SS1D
import sys

class KANInverseModel(nn.Module):
    def __init__(self, gru_hidden=8, dropout=0.0, nonlinearity='tanh'):
        super().__init__()
        act = nn.ReLU() if nonlinearity == "relu" else nn.Tanh()
        self.act = act
        self.dropout = nn.Dropout(dropout)

        self.cnn1 = nn.Conv2d(1, 8, kernel_size=(1, 5), padding=(0, 2), dilation=(1, 1))
        self.gn1 = nn.GroupNorm(1, 8)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.cnn2 = nn.Conv2d(1, 8, kernel_size=(1, 5), padding=(0, 6), dilation=(1, 3))
        self.gn2 = nn.GroupNorm(1, 8)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.cnn3 = nn.Conv2d(1, 8, kernel_size=(1, 5), padding=(0, 12), dilation=(1, 6))
        self.gn3 = nn.GroupNorm(1, 8)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.cnn4 = nn.Conv2d(24, 16, (3, 3), padding=(0, 1))
        self.gn4  = nn.GroupNorm(1, 16)
        self.cnn5 = nn.Conv2d(16, 16, (3, 3), padding=(0, 1))
        self.gn5  = nn.GroupNorm(1, 16)
        self.cnn6 = nn.Conv2d(16, 16, 1)
        self.gn6  = nn.GroupNorm(1, 16)

        self.gru = nn.GRU(1, gru_hidden, num_layers=3,
                          bidirectional=True, batch_first=True)

        self.cnnt1 = nn.ConvTranspose1d(16, 8, 4, stride=2, padding=1)
        self.gn7   = nn.GroupNorm(1, 8)
        self.cnnt2 = nn.ConvTranspose1d(8, 8, 4, stride=2, padding=1)
        self.gn8   = nn.GroupNorm(1, 8)
        self.kan_norm = nn.LayerNorm(16)    #归一化
        self.fusion_conv = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=1, bias=False)  #自适应融合

        self.fusion_norm = nn.GroupNorm(1, 16)  # 或 nn.BatchNorm1d(16)
        self.fusion_act = nn.ReLU() if nonlinearity == 'relu' else nn.Tanh()

        self.gru_out = nn.GRU(8, 8, num_layers=1,
                              batch_first=True, bidirectional=True)

        self.out = KAN(
            layers_hidden=[16, 8, 1],
            grid_size=3,
            spline_order=2,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-10, 10],
        )

        # ---------- 参数初始化 ----------
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d)):
                if m.weight is not None:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                if m.weight is not None:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # ⚠️ 注意：KAN 里面自己 reset_parameters() 了，不需要你这里手动初始化它

    # x:(B,1,470);  x_expand:(B,1,5,470)
    def forward(self, x, x_expand,update_grid=False):
        # 三路卷积分支
        f1 = self.pool1(self.gn1(self.cnn1(x_expand)))
        f2 = self.pool2(self.gn2(self.cnn2(x_expand)))
        f3 = self.pool3(self.gn3(self.cnn3(x_expand)))
        feat = torch.cat([f1, f2, f3], dim=1)        # [B,24,5,T]

        feat = self.act(self.gn4(self.cnn4(feat)))
        feat = self.act(self.gn5(self.cnn5(feat)))
        feat = self.act(self.gn6(self.cnn6(feat)))   # [B,16,5,T]

        feat = self.act(feat).squeeze(2)            # [B,16,T]

        rnn_out, _ = self.gru(x.transpose(-1, -2))   # 输入 [B,T,1]
        rnn_out = rnn_out.transpose(-1, -2)          # [B,16,T]

        z = torch.cat([rnn_out, feat], dim=1)  # [B,32,T]
        z = self.fusion_conv(z)  # 1×1 conv 32→16
        z = self.fusion_norm(z)
        z = self.fusion_act(z)

        z = self.act(self.gn7(self.cnnt1(z)))        # [B,8,T*2]
        z = self.act(self.gn8(self.cnnt2(z)))        # [B,8,T*4]
        z, _ = self.gru_out(z.transpose(-1, -2))     # z: [B,T',16]

        B, T, C = z.shape                             # C=16
        z2 = z.reshape(B * T, C)# [B*T,16]
        z2 = self.kan_norm(z2)
        y2 = self.out(z2,update_grid=update_grid)                             # [B*T,1]
        y  = y2.reshape(B, T, 1)                      # [B,T,1]

        return y.transpose(-1, -2)                    # [B,1,T]
#

class KANForwardModel(nn.Module):
    def __init__(self, resolution_ratio=4, nonlinearity="tanh"):
        super(KANForwardModel, self).__init__()
        self.resolution_ratio = resolution_ratio
        self.activation = nn.ReLU() if nonlinearity == "relu" else nn.Tanh()
        self.cnn1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=9, padding=4)
        self.cnn2 = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=7, padding=3)
        self.cnn3 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3, padding=1)
        self.wavelet = nn.Conv1d(in_channels=1, out_channels=1, stride=self.resolution_ratio, kernel_size=50,
                                 padding=int((50 - self.resolution_ratio + 2) / 2))

    def forward(self, x):
        x = self.cnn1(x)
        x = self.activation(x)
        x = self.cnn2(x)
        x = self.activation(x)
        x = self.cnn3(x)
        x = self.wavelet(x)
        return x