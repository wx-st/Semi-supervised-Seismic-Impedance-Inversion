import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn.functional import conv1d
import pywt
import torch.nn.functional as F
import numpy as np
from pytorch_wavelets import DWTForward

class inverse_model(nn.Module):
    def __init__(self, gru_hidden=8, dropout=0.0, nonlinearity='tanh'):
        super().__init__()
        act = nn.ReLU() if nonlinearity == "relu" else nn.Tanh()
        self.act = act
        self.dropout = nn.Dropout(dropout)

        # === 多尺度分支（唯一结构差异） ===

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 5), dilation=(1, 1), padding=(0, 2))
        self.gn1 = nn.GroupNorm(1, 8)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.cnn2 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 5), dilation=(1, 3), padding=(0, 6))
        self.gn2 = nn.GroupNorm(1, 8)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.cnn3 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 5), dilation=(1, 6), padding=(0, 12))
        self.gn3 = nn.GroupNorm(1, 8)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # -------- 深层卷积  --------
        self.cnn4 = nn.Conv2d(24, 16, (3,3), padding=(0,1))
        self.gn4  = nn.GroupNorm(1,16)
        self.cnn5 = nn.Conv2d(16, 16, (3,3), padding=(0,1))
        self.gn5  = nn.GroupNorm(1,16)
        self.cnn6 = nn.Conv2d(16, 16, 1)
        self.gn6  = nn.GroupNorm(1,16)


        self.gru = nn.GRU(1, gru_hidden, num_layers=3,
                          bidirectional=True, batch_first=True)
        self.cnnt1 = nn.ConvTranspose1d(16, 8, 4, stride=2, padding=1)
        self.gn7   = nn.GroupNorm(1,8)
        self.cnnt2 = nn.ConvTranspose1d(8, 8, 4, stride=2, padding=1)
        self.gn8   = nn.GroupNorm(1,8)
        self.gru_out = nn.GRU(8, 8, num_layers=1,
                              batch_first=True, bidirectional=True)
        self.out = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.act = act

        # ---------- 参数初始化 ----------
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d)):
                if m.weight is not None:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:  # ← 加这一层判断
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                if m.weight is not None:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # x:(B,1,470);  x_expand:(B,1,5,470)
    def forward(self, x, x_expand):
        # 三路卷积分支
        f1 = self.pool1(self.gn1(self.cnn1(x_expand)))
        f2 = self.pool2(self.gn2(self.cnn2(x_expand)))
        f3 = self.pool3(self.gn3(self.cnn3(x_expand)))
        feat = torch.cat([f1,f2,f3], dim=1)        # [B,24,5,T]
        feat = self.act(self.gn4(self.cnn4(feat)))
        feat = self.act(self.gn5(self.cnn5(feat)))
        feat = self.act(self.gn6(self.cnn6(feat)))  # [B,16,5,T]
        feat = self.act(feat).squeeze(2)           # [B,16,T]
        rnn_out,_ = self.gru(x.transpose(-1,-2))
        rnn_out   = rnn_out.transpose(-1,-2)
        z = rnn_out + feat

        z = self.act(self.gn7(self.cnnt1(z)))
        z = self.act(self.gn8(self.cnnt2(z)))
        z,_ = self.gru_out(z.transpose(-1,-2))

        return self.out(z).transpose(-1,-2)


class forward_model(nn.Module):
    def __init__(self,resolution_ratio=4, nonlinearity="tanh"):
        super(forward_model, self).__init__()
        self.resolution_ratio = resolution_ratio
        self.activation = nn.ReLU() if nonlinearity == "relu" else nn.Tanh()
        self.cnn1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=9, padding=4)
        self.cnn2 = nn.Conv1d(in_channels=4, out_channels=4,kernel_size=7, padding=3)
        self.cnn3 = nn.Conv1d(in_channels=4, out_channels=1,kernel_size=3, padding=1)

        self.wavelet = nn.Conv1d(in_channels=1, out_channels=1, stride=self.resolution_ratio, kernel_size=50,padding=int((50-self.resolution_ratio+2)/2))

    def forward(self, x):
        x = self.cnn1(x)
        x = self.activation(x)
        x = self.cnn2(x)
        x = self.activation(x)
        x = self.cnn3(x)
        x = self.wavelet(x)
        return x