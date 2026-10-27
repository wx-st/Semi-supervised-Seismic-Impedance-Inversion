import torch
import torch.nn as nn
import torch.nn.functional as F
from kan import KAN

class SEBlock2D(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock2D, self).__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.avg_pool(x)   # [B,C,1,1]
        w = self.fc(w)         # [B,C,1,1]
        return x * w

class KANInverseModel(nn.Module):
    """
    多道输入、多道输出版
    输入:
        x_win: [B, 1, in_width, T_low]
    输出:
        y_pred: [B, out_width, T_high]
    """
    def __init__(self, in_width=9, out_width=5, gru_hidden=8, dropout=0.0, nonlinearity='tanh'):
        super().__init__()
        assert out_width <= in_width
        assert in_width % 2 == 1
        assert out_width % 2 == 1

        self.in_width = in_width
        self.out_width = out_width
        self.mid_left = (in_width - out_width) // 2
        self.mid_right = self.mid_left + out_width

        act = nn.ReLU() if nonlinearity == "relu" else nn.Tanh()
        self.act = act
        self.dropout = nn.Dropout(dropout)

        # ===== 横向+纵向联合特征提取：保留多尺度2D CNN =====
        self.cnn1 = nn.Conv2d(1, 8, kernel_size=(1, 5), padding=(0, 2), dilation=(1, 1))
        self.gn1 = nn.GroupNorm(1, 8)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.cnn2 = nn.Conv2d(1, 8, kernel_size=(1, 5), padding=(0, 6), dilation=(1, 3))
        self.gn2 = nn.GroupNorm(1, 8)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.cnn3 = nn.Conv2d(1, 8, kernel_size=(1, 5), padding=(0, 12), dilation=(1, 6))
        self.gn3 = nn.GroupNorm(1, 8)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.cnn4 = nn.Conv2d(24, 16, (3, 3), padding=(1, 1))
        self.gn4  = nn.GroupNorm(1, 16)
        self.cnn5 = nn.Conv2d(16, 16, (3, 3), padding=(1, 1))
        self.gn5  = nn.GroupNorm(1, 16)
        self.se = SEBlock2D(channels=16, reduction=8)
        self.cnn6 = nn.Conv2d(16, 16, 1)
        self.gn6  = nn.GroupNorm(1, 16)

        # ===== 对输出窗口中的每条目标道分别做GRU编码 =====
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=gru_hidden,
            num_layers=3,
            bidirectional=True,
            batch_first=True
        )

        # 融合后升采样
        self.fusion_conv = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=1, bias=False)
        self.fusion_norm = nn.GroupNorm(1, 16)
        self.fusion_act = nn.ReLU() if nonlinearity == 'relu' else nn.Tanh()

        self.cnnt1 = nn.ConvTranspose1d(16, 8, 4, stride=2, padding=1)
        self.gn7   = nn.GroupNorm(1, 8)
        self.cnnt2 = nn.ConvTranspose1d(8, 8, 4, stride=2, padding=1)
        self.gn8   = nn.GroupNorm(1, 8)

        self.gru_out = nn.GRU(
            input_size=8,
            hidden_size=8,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.kan_norm = nn.LayerNorm(16)

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

        # 参数初始化
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

    def forward(self, x_win, update_grid=False):
        """
        x_win: [B, 1, in_width, T_low]
        return: [B, out_width, T_high]
        """
        B, C, W, T = x_win.shape
        assert W == self.in_width

        # ========= 1) 多尺度2D CNN提取整个窗口特征 =========
        f1 = self.pool1(self.gn1(self.cnn1(x_win)))
        f2 = self.pool2(self.gn2(self.cnn2(x_win)))
        f3 = self.pool3(self.gn3(self.cnn3(x_win)))
        feat = torch.cat([f1, f2, f3], dim=1)          # [B,24,W,T]

        feat = self.act(self.gn4(self.cnn4(feat)))     # [B,16,W,T]
        feat = self.act(self.gn5(self.cnn5(feat))) # [B,16,W,T]
        feat = self.se(feat)
        feat = self.act(self.gn6(self.cnn6(feat)))     # [B,16,W,T]

        # 只取中间 out_width 条目标道对应的 CNN 特征
        feat = feat[:, :, self.mid_left:self.mid_right, :]   # [B,16,out_width,T]

        # ========= 2) 输出窗口中的每条目标道各自做GRU =========
        x_targets = x_win[:, :, self.mid_left:self.mid_right, :]   # [B,1,out_width,T]
        x_targets = x_targets.squeeze(1)                           # [B,out_width,T]

        # reshape成 (B*out_width, T, 1)
        x_rnn = x_targets.reshape(B * self.out_width, T, 1)
        rnn_out, _ = self.gru(x_rnn)                               # [B*out_width, T, 16]
        rnn_out = rnn_out.transpose(1, 2)                          # [B*out_width, 16, T]
        rnn_out = rnn_out.reshape(B, self.out_width, 16, T).permute(0, 2, 1, 3)  # [B,16,out_width,T]

        # ========= 3) CNN特征 + GRU特征 融合 =========
        z = torch.cat([feat, rnn_out], dim=1)                      # [B,32,out_width,T]

        # 对每一条目标道单独做1D融合与反演
        z = z.permute(0, 2, 1, 3).reshape(B * self.out_width, 32, T)   # [B*out_width,32,T]
        z = self.fusion_conv(z)                                     # [B*out_width,16,T]
        z = self.fusion_norm(z)
        z = self.fusion_act(z)

        z = self.act(self.gn7(self.cnnt1(z)))                       # [B*out_width,8,T*2]
        z = self.act(self.gn8(self.cnnt2(z)))                       # [B*out_width,8,T*4]

        z, _ = self.gru_out(z.transpose(1, 2))                      # [B*out_width,T_high,16]

        B2, T2, C2 = z.shape
        z2 = z.reshape(B2 * T2, C2)
        z2 = self.kan_norm(z2)
        y2 = self.out(z2, update_grid=update_grid)                  # [B*out_width*T_high,1]

        y = y2.reshape(B, self.out_width, T2)                       # [B,out_width,T_high]
        return y

class KANForwardModel(nn.Module):
    def __init__(self, resolution_ratio=4, nonlinearity="tanh"):
            super(KANForwardModel, self).__init__()
            self.resolution_ratio = resolution_ratio
            self.activation = nn.ReLU() if nonlinearity == "relu" else nn.Tanh()

            self.cnn1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=9, padding=4)
            self.cnn2 = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=7, padding=3)
            self.cnn3 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3, padding=1)

            self.wavelet = nn.Conv1d(
                in_channels=1,
                out_channels=1,
                stride=self.resolution_ratio,
                kernel_size=50,
                padding=int((50 - self.resolution_ratio + 2) / 2)
            )

    def forward(self, x):
            """
            x:
                [B, out_width, T_high] 或 [B,1,T_high]
            return:
                [B, out_width, T_low] 或 [B,1,T_low]
            """
            if x.dim() != 3:
                raise ValueError(f"Expected x dim=3, got {x.shape}")

            B, W, T = x.shape
            x = x.reshape(B * W, 1, T)

            x = self.cnn1(x)
            x = self.activation(x)
            x = self.cnn2(x)
            x = self.activation(x)
            x = self.cnn3(x)
            x = self.wavelet(x)  # [B*W,1,T_low]

            T_low = x.shape[-1]
            x = x.reshape(B, W, T_low)
            return x