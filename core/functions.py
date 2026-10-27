import numpy as np
import torch
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from skimage.metrics import structural_similarity as _ssim
from skimage.metrics import peak_signal_noise_ratio as _psnr

class Normalization:
    def __init__(self, mean_val=None, std_val=None):
        self.mean_val = mean_val
        self.std_val = std_val

    def normalize(self, x):
        return (x-self.mean_val)/self.std_val

    def unnormalize(self, x):
        return x*self.std_val + self.mean_val


def metrics(y, x):
    # y: pred, x: true（保持你原来的形参顺序）
    import numpy as np
    import torch

    # to numpy float64
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)

    # 用 NaN 作为无效：只在两者都有限的地方保留
    valid = np.isfinite(x) & np.isfinite(y)
    x_ = np.where(valid, x, np.nan)
    y_ = np.where(valid, y, np.nan)

    # 沿最后一维统计（保持 keepdims=True，与原代码对齐）
    x_mean = np.nanmean(x_, axis=-1, keepdims=True)
    y_mean = np.nanmean(y_, axis=-1, keepdims=True)
    x_std  = np.nanstd (x_, axis=-1, keepdims=True)
    y_std  = np.nanstd (y_, axis=-1, keepdims=True)

    # 相关系数：零方差保护
    num   = np.nanmean((x_ - x_mean) * (y_ - y_mean), axis=-1, keepdims=True)
    denom = x_std * y_std
    corr  = np.where(denom > 0, num / denom, 0.0)

    # R^2：常数标签保护
    S_tot = np.nansum((x_ - x_mean) ** 2, axis=-1, keepdims=True)
    S_res = np.nansum((x_ - y_)      ** 2, axis=-1, keepdims=True)
    r2    = np.where(S_tot > 0, 1.0 - S_res / S_tot, 0.0)

    return torch.tensor(corr, dtype=torch.float32), torch.tensor(r2, dtype=torch.float32)

def total_variation_loss(y):
    """
    y: (B, D, 3) 或 (B, T, 3)，表示 (depth, param)
    平滑度先验损失，鼓励 y 在 D 维度上平滑
    """
    dy = torch.abs(y[:, 1:, :] - y[:, :-1, :])  # (B, D-1, 3)
    return dy.mean()

def display_results(loss, property_corr, property_r2, args, header):
    property_corr = torch.mean(torch.cat(property_corr), dim=0).squeeze()
    property_r2 = torch.mean(torch.cat(property_r2), dim=0).squeeze()
    loss = torch.mean(torch.tensor(loss))
    print("loss: {:.4f}\nCorrelation: {:0.4f}\nr2 Coeff.  : {:0.4f}".format(loss, property_corr, property_r2))


def PSNR(true_image, real_image):
    """
    支持 (B, T) 或 (T,)；内部做掩膜与零范围保护；返回 batch 的 nanmean 标量
    """
    import numpy as np
    import torch

    # to numpy float64
    if torch.is_tensor(true_image):
        true_image = true_image.detach().cpu().numpy()
    if torch.is_tensor(real_image):
        real_image = real_image.detach().cpu().numpy()
    t = true_image.astype(np.float64, copy=False)
    p = real_image.astype(np.float64, copy=False)

    def _psnr_1d(tt, pp):
        mask = np.isfinite(tt) & np.isfinite(pp)
        if mask.sum() < 2:
            return np.nan
        tt = tt[mask]; pp = pp[mask]
        dr = tt.max() - tt.min()
        if not np.isfinite(dr) or dr <= 0:
            return np.nan
        mse = np.mean((tt - pp) ** 2)
        if mse <= 0:
            return np.inf
        return 20.0 * np.log10(dr) - 10.0 * np.log10(mse)

    if t.ndim == 1:
        return _psnr_1d(t, p)
    elif t.ndim == 2:
        vals = [ _psnr_1d(t[i], p[i]) for i in range(t.shape[0]) ]
        return float(np.nanmean(vals))
    else:
        # 其他形状：展平到最后一维再按 batch 处理
        t2 = t.reshape(-1, t.shape[-1])
        p2 = p.reshape(-1, p.shape[-1])
        vals = [ _psnr_1d(t2[i], p2[i]) for i in range(t2.shape[0]) ]
        return float(np.nanmean(vals))



# SSIM（逐 trace；共同有效区；极差/窗口保护）
# =========================
def SSIM(true_image, pred_image, reduce: str = "mean"):
    """
    逐 trace SSIM：只在 GT/Pred 的“共同 finite 区域”上算；
    自动避开极差≈0/长度过短的 trace；返回有效 trace 的 nanmean（或 reduce='none' 返回每条）。
    支持形状：(T,), (B,T), (N,1,T), (N,T,1), (N,T)
    """
    def _to_np(a):
        if torch.is_tensor(a):
            a = a.detach().cpu().numpy()
        return np.asarray(a, dtype=np.float32)

    t = np.squeeze(_to_np(true_image))
    p = np.squeeze(_to_np(pred_image))

    # 对齐形状（最后一维视作长度维）
    if t.shape != p.shape:
        t = t.reshape(-1, t.shape[-1])
        p = p.reshape(-1, p.shape[-1])
    if t.ndim == 1:
        t = t[None, :]
        p = p[None, :]

    vals = []
    for i in range(t.shape[0]):
        tt = t[i]
        pp = p[i]
        m = np.isfinite(tt) & np.isfinite(pp)
        if m.sum() < 3:
            vals.append(np.nan); continue
        tt = tt[m]; pp = pp[m]

        dr = float(tt.max() - tt.min())
        if not np.isfinite(dr) or dr <= 0:
            vals.append(np.nan); continue  # 极差≈0 的 trace 跳过

        L = tt.shape[0]
        win = min(11, L)            # 窗口必须奇数
        if win % 2 == 0:
            win -= 1
        if win < 3:
            vals.append(np.nan); continue

        try:
            s = _ssim(
                tt[:, None], pp[:, None],
                data_range=dr,
                gaussian_weights=True,
                use_sample_covariance=False,
                win_size=win,
                channel_axis=None     # 等价于 multichannel=False
            )
        except Exception:
            s = np.nan
        vals.append(s)

    vals = np.array(vals, dtype=np.float32)
    if reduce == "none":
        return vals
    # 全无效则返回 0.0，避免打印 nan
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else 0.0



def accuracy(predict_mean, predict_std, true_image):
    lower_bound = predict_mean - 2 * predict_std
    upper_bound = predict_mean + 2 * predict_std
    flag = (true_image > lower_bound) * (true_image < upper_bound)
    accuracy = np.sum(flag) / (flag.shape[0] * flag.shape[1])

    return accuracy
























