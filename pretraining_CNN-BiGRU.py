import argparse
import os, random

# ========= Reproducibility (use your original random_seed name) =========
random_seed = 30  # 保持原名不变
os.environ["PYTHONHASHSEED"] = str(random_seed)
# 让 CUBLAS 在确定性模式下工作（需在 import torch 前设置）
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# =======================================================================

import torch
from os.path import isdir
from models.CNN_BiGRU  import inverse_model, forward_model
from torch.utils import data
from core.functions import *
from torch import nn, optim
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import csv
import torch.nn.functional as F
# 在 import torch 后添加
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 然后修改所有 .cuda() 调用为 .to(device)
def _ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

@torch.no_grad()
def _batch_corr(yhat, y, eps=1e-8):
    # yhat, y: [B, 1, T] or [B, T]
    if yhat.dim() == 3: yhat = yhat.squeeze(1)
    if y.dim() == 3: y = y.squeeze(1)
    x = yhat - yhat.mean(dim=-1, keepdim=True)
    z = y - y.mean(dim=-1, keepdim=True)
    num = (x * z).sum(dim=-1)
    den = torch.sqrt((x * x).sum(dim=-1) * (z * z).sum(dim=-1) + eps)
    corr = num / (den + eps)                 # [B]
    return corr.mean()                       # 标量

@torch.no_grad()
def _batch_r2(yhat, y, eps=1e-8):
    # yhat, y: [B, 1, T] or [B, T]
    if yhat.dim() == 3: yhat = yhat.squeeze(1)
    if y.dim() == 3: y = y.squeeze(1)
    y_mean = y.mean(dim=-1, keepdim=True)
    sse = ((y - yhat) ** 2).sum(dim=-1)      # [B]
    sst = ((y - y_mean) ** 2).sum(dim=-1) + eps
    r2 = 1.0 - sse / sst
    return r2.mean()                         # 标量

def set_seed_all(seed: int = 30, deterministic: bool = True):
    """一次性固定 Python/NumPy/PyTorch/CUDA 的随机性，并返回一个可复现的 DataLoader 生成器。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # 关闭 TF32，避免在 Ampere+ 上的数值差异
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass
        # 使用确定性算子（不支持时仅告警）
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g

# 统一设置一次，并给 DataLoader 复用
torch_generator = set_seed_all(random_seed, deterministic=True)

warnings.filterwarnings('ignore')

def get_data(args, test=False):

    data_dic = np.load("data/data.npy", allow_pickle=True).item()

    seismic_data = data_dic["seismic"]
    acoustic_impedance_data = data_dic["acoustic_impedance"]

    seismic_mean = torch.tensor(np.mean(seismic_data, keepdims=True)).float()
    seismic_std = torch.tensor(np.std(seismic_data, keepdims=True)).float()

    acoustic_mean= torch.tensor(np.mean(acoustic_impedance_data, keepdims=True)).float()
    acoustic_std = torch.tensor(np.std(acoustic_impedance_data, keepdims=True)).float()

    seismic_data = torch.tensor(seismic_data).float()
    acoustic_impedance_data = torch.tensor(acoustic_impedance_data).float()

    seismic_data = seismic_data.to(device)
    acoustic_impedance_data = acoustic_impedance_data.to(device)

    seismic_mean = seismic_mean.to(device)
    seismic_std = seismic_std.to(device)

    acoustic_mean = acoustic_mean.to(device)
    acoustic_std = acoustic_std.to(device)

    seismic_normalization = Normalization(mean_val=seismic_mean, std_val=seismic_std)
    acoustic_normalization = Normalization(mean_val=acoustic_mean, std_val=acoustic_std)

    seismic_data = seismic_normalization.normalize(seismic_data)
    acoustic_impedance_data = acoustic_normalization.normalize(acoustic_impedance_data)

    # expand seismic_data
    # seismic_data_fill:(N+args.width-1, 1, T), args.width has to be odd
    seismic_data_fill = torch.zeros(seismic_data.shape[0]+args.width-1, seismic_data.shape[1], seismic_data.shape[2]).to(device)
    seismic_data_fill[int((args.width-1)/2):int((args.width-1)/2)+seismic_data.shape[0], :] = seismic_data
    # seismic_data_expand:(N, 1, args.width, T)
    seismic_data_expand = torch.zeros(seismic_data.shape[0], seismic_data.shape[1], args.width, seismic_data.shape[2]).to(device)
    for i in range(seismic_data.shape[0]):
        seismic_data_expand[i, :] = seismic_data_fill[i:i+args.width, :].transpose(0, 1)
    # seismic_data:(N+args.width-1, 1, args.width+1, 470)
    seismic_data = torch.cat((seismic_data_expand, seismic_data.unsqueeze(2)), dim=2)

    if not test:
        num_samples = seismic_data.shape[0]
        indecies = np.arange(0, num_samples)
        train_indecies = indecies[(np.linspace(0, len(indecies)-1, args.num_train_wells)).astype(int)]

        train_data = data.Subset(data.TensorDataset(seismic_data, acoustic_impedance_data), train_indecies)
        train_loader = data.DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=False,  # 保持你的原设定
            generator=torch_generator,
            worker_init_fn=lambda wid: np.random.seed(random_seed + wid),
            num_workers=0,
            pin_memory=False
        )

        unlabeled_loader = data.DataLoader(
            data.TensorDataset(seismic_data),
            batch_size=args.batch_size,
            shuffle=True,   # 保持你的原设定
            generator=torch_generator,
            worker_init_fn=lambda wid: np.random.seed(random_seed + wid),
            num_workers=0,
            pin_memory=False
        )
        return train_loader, unlabeled_loader, seismic_normalization, acoustic_normalization
    else:
        test_loader = data.DataLoader(
            data.TensorDataset(seismic_data, acoustic_impedance_data),
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            generator=torch_generator,
            worker_init_fn=lambda wid: np.random.seed(random_seed + wid),
            num_workers=0,
            pin_memory=False
        )
        return test_loader, seismic_normalization, acoustic_normalization


def get_models(args):
    if args.test_checkpoint is None:
        inverse_net = inverse_model(nonlinearity=args.nonlinearity)
        forward_net = forward_model(nonlinearity=args.nonlinearity)
        optimizer = optim.Adam(list(inverse_net.parameters())+list(forward_net.parameters()), amsgrad=True, lr=0.001)
    else:
        try:
            inverse_net = torch.load("checkpoints/" + args.test_checkpoint + "_inverse", weights_only=False)
            forward_net = torch.load("checkpoints/" + args.test_checkpoint + "_forward", weights_only=False)
            optimizer = torch.load("checkpoints/" + args.test_checkpoint + "_optimizer", weights_only=False)

        except FileNotFoundError:
            try:
                inverse_net = torch.load(args.test_checkpoint + "_inverse")
                forward_net = torch.load(args.test_checkpoint + "_forward")
                optimizer = torch.load(args.test_checkpoint + "_optimizer")
            except:
                print("No checkpoint found at '{}'- Please specify the model for testing".format(args.test_checkpoint))
                exit()

    inverse_net.to(device)
    forward_net.to(device)

    return inverse_net, forward_net, optimizer


def train(args):
    train_loader, unlabeled_loader, seismic_normalization, acoustic_normalization = get_data(args)
    inverse_net, forward_net, optimizer = get_models(args)
    inverse_net.train()
    criterion = nn.MSELoss()

    if not isdir("checkpoints"):
        os.mkdir("checkpoints")

    # === 日志文件（按 session_name 区分）===
    log_path = os.path.join("logs", f"{args.session_name}_train_metrics.csv")
    _ensure_dir(log_path)
    # 若不存在则写表头
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch","train_loss","property_loss","seismic_loss","mse_y","mse_x","r2","corr"])

    loss_var = []
    print("Training the model")
    for epoch in tqdm(range(args.max_epoch)):
        batch_losses = []

        # 这几个用来做 epoch 平均
        ep_prop_losses = []
        ep_seis_losses = []
        ep_mse_y = []
        ep_mse_x = []
        ep_r2 = []
        ep_corr = []

        for x, y in train_loader:
            optimizer.zero_grad()

            x_expand = x[:, :, 0:args.width]
            x = x[:, :, args.width]          # [B,1,T]
            y_pred = inverse_net(x, x_expand)
            x_rec  = forward_net(y)

            # property / seismic loss
            property_loss = criterion(y_pred, y) + criterion(x_rec, x)

            if args.beta != 0:
                try:
                    x_u = next(unlabeled)[0]
                except:
                    unlabeled = iter(unlabeled_loader)
                    x_u = next(unlabeled)[0]
                x_u_expand = x_u[:, :, 0:args.width]
                x_u = x_u[:, :, args.width]
                y_u_pred = inverse_net(x_u, x_u_expand)
                x_u_rec = forward_net(y_u_pred)
                seismic_loss = criterion(x_u_rec, x_u)
            else:
                seismic_loss = torch.tensor(0.0, device=y.device)

            loss = args.alpha * property_loss + args.beta * seismic_loss
            loss.backward()
            optimizer.step()

            # —— 收集 batch 级指标（detach）——
            with torch.no_grad():
                mse_y = F.mse_loss(y_pred, y)
                mse_x = F.mse_loss(x_rec, x)
                r2    = _batch_r2(y_pred, y)
                corr  = _batch_corr(y_pred, y)

                ep_prop_losses.append(property_loss.detach().item())
                ep_seis_losses.append(seismic_loss.detach().item())
                ep_mse_y.append(mse_y.detach().item())
                ep_mse_x.append(mse_x.detach().item())
                ep_r2.append(r2.detach().item())
                ep_corr.append(corr.detach().item())

            batch_losses.append(loss.detach().item())

        # —— epoch 平均 ——
        train_loss = float(np.mean(batch_losses))
        loss_var.append(train_loss)

        mean_prop = float(np.mean(ep_prop_losses))
        mean_seis = float(np.mean(ep_seis_losses))
        mean_mse_y = float(np.mean(ep_mse_y))
        mean_mse_x = float(np.mean(ep_mse_x))
        mean_r2 = float(np.mean(ep_r2))
        mean_corr = float(np.mean(ep_corr))

        # 写入 CSV
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, mean_prop, mean_seis, mean_mse_y, mean_mse_x, mean_r2, mean_corr])

    loss_var = torch.tensor(loss_var).cpu().numpy()

    # 保存目录：和 test() 一样放到 results/<session_name>/
    run_dir = os.path.join("results", args.session_name)
    os.makedirs(run_dir, exist_ok=True)

    # （可选）保存loss数值，方便之后复现/画图
    np.save(os.path.join(run_dir, "train_loss.npy"), loss_var)

    # 保存loss曲线图
    fig = plt.figure()
    plt.plot(np.arange(loss_var.shape[0]), loss_var)
    plt.title("Curve of Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    fig.savefig(os.path.join(run_dir, "train_loss.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Training loss curve saved to {os.path.join(run_dir, 'train_loss.png')}")

    torch.save(inverse_net, "checkpoints/{}_inverse".format(args.session_name))
    torch.save(forward_net, "checkpoints/{}_forward".format(args.session_name))
    torch.save(optimizer, "checkpoints/{}_optimizer".format(args.session_name))


def test(args):
    # === 数据与模型 ===
    test_loader, seismic_normalization, acoustic_normalization = get_data(args, test=True)
    if args.test_checkpoint is None:
        args.test_checkpoint = args.session_name
    inverse_net, forward_net, _ = get_models(args)
    criterion = nn.MSELoss(reduction="sum")

    # === 固定输出目录 ===
    run_dir = os.path.join("results", args.session_name)   # results/<session_name>/
    os.makedirs(run_dir, exist_ok=True)

    # === 收集容器 ===
    predicted_impedance, true_impedance = [], []
    predicted_seismic_list, true_seismic_list = [], []
    test_property_corr, test_property_r2 = [], []

    inverse_net.eval()
    print("\nTesting the model\n")

    with torch.no_grad():
        test_loss = []
        for x, y in test_loader:
            x_expand = x[:, :, 0:args.width]
            x_center = x[:, :, args.width]

            y_pred = inverse_net(x_center, x_expand)
            x_rec  = forward_net(y_pred)

            loss = criterion(y_pred, y) / np.prod(y.shape)
            test_loss.append(loss.item())

            corr, r2 = metrics(y_pred.detach(), y.detach())
            test_property_corr.append(corr)
            test_property_r2.append(r2)

            true_impedance.append(y)
            predicted_impedance.append(y_pred)
            true_seismic_list.append(x_center)
            predicted_seismic_list.append(x_rec)

        # === 汇总指标 ===
        property_corr = torch.mean(torch.cat(test_property_corr), dim=0).squeeze()
        property_r2   = torch.mean(torch.cat(test_property_r2),   dim=0).squeeze()
        loss_scalar   = torch.mean(torch.tensor(test_loss)).item()
        print(f"loss: {loss_scalar:.4f}\nCorrelation: {property_corr:.4f}\nr2 Coeff.: {property_r2:.4f}")

        # === 拼接 & 反归一化 ===
        predicted_impedance = acoustic_normalization.unnormalize(torch.cat(predicted_impedance, dim=0))
        true_impedance      = acoustic_normalization.unnormalize(torch.cat(true_impedance,      dim=0))
        true_seis           = seismic_normalization.unnormalize(torch.cat(true_seismic_list,    dim=0))
        pred_seis           = seismic_normalization.unnormalize(torch.cat(predicted_seismic_list, dim=0))

        # === 转 numpy ===
        predicted_impedance = predicted_impedance.cpu().numpy()
        true_impedance      = true_impedance.cpu().numpy()
        true_seis           = true_seis.cpu().numpy()
        pred_seis           = pred_seis.cpu().numpy()
        # ========= 2D 剖面图（True / Pred / Error）=========
        # 形状: [N, 1, T] -> [N, T]
        true_sec = true_seis[:, 0, :]
        pred_sec = pred_seis[:, 0, :]
        err_sec  = pred_sec - true_sec

        def _save_section(img, title, fname, vmin=None, vmax=None, cmap='seismic'):
            fig = plt.figure(figsize=(12, 6))
            plt.imshow(
                img.T,
                aspect='auto',
                origin='upper',
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                interpolation="nearest"
            )
            plt.title(title)
            plt.xlabel("Trace Index")
            plt.ylabel("Time Sample")
            plt.colorbar()
            fig.savefig(os.path.join(run_dir, fname), dpi=300, bbox_inches='tight', pad_inches=0.05)
            plt.close(fig)



            # 对称裁剪（99% 分位），让颜色分布更像论文里的地震剖面
        clip = np.percentile(np.abs(true_sec), 99)  # 你也可以用 98/99.5 看效果
        vmin, vmax = -clip, clip
        _save_section(true_sec, "True Seismic Section", "section_true_seis.png",
              vmin=vmin, vmax=vmax, cmap="seismic")

        _save_section(pred_sec, "Predicted Seismic Section", "section_pred_seis.png",
              vmin=vmin, vmax=vmax, cmap="seismic")

# 误差一般也用发散色标更直观（围绕0）
        eclip = np.percentile(np.abs(err_sec), 99)
        _save_section(err_sec, "Error Seismic Section (Pred-True)", "section_err_seis.png",
              vmin=-eclip, vmax=eclip, cmap="seismic")

        # （可选）阻抗剖面图
        true_ai = true_impedance[:, 0, :]
        pred_ai = predicted_impedance[:, 0, :]
        err_ai = np.abs(pred_ai - true_ai)

        # 真实/预测波阻抗
        ai_vmin = np.min(true_ai)
        ai_vmax = np.max(true_ai)

        _save_section(
            true_ai,
            "True Acoustic Impedance Section",
            "section_true_ai.png",
            vmin=ai_vmin,
            vmax=ai_vmax,
            cmap='jet'
        )

        _save_section(
            pred_ai,
            "Predicted Acoustic Impedance Section",
            "section_pred_ai.png",
            vmin=ai_vmin,
            vmax=ai_vmax,
            cmap='jet'
        )



        err_vmin = 0.0
        err_vmax = 7000

        _save_section(
            err_ai,
            "Absolute Error AI Section",
            "section_err_ai.png",
            vmin=err_vmin,
            vmax=err_vmax,
            cmap='binary'  # 黑底灰白突出
        )

        # === 指标 ===
        psnr = PSNR(true_impedance[:, 0], predicted_impedance[:, 0])

        print(f"PSNR: {psnr:.4f}\n")

        # === 保存指标 ===
        import json
        with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
            json.dump({
                "loss": float(loss_scalar),
                "corr": float(property_corr),
                "r2":   float(property_r2),
                "psnr": float(psnr),

                "session_name": args.session_name
            }, f, indent=2)

        # === 保存数组 ===
        np.save(os.path.join(run_dir, "true_impedance.npy"), true_impedance)
        np.save(os.path.join(run_dir, "pred_impedance.npy"), predicted_impedance)
        np.save(os.path.join(run_dir, "true_seis.npy"), true_seis)
        np.save(os.path.join(run_dir, "pred_seis.npy"), pred_seis)

        # === 固定剖面选择 ===
        trace_groups = [
            [625, 1250, 1875],   # 第一组
            [300, 900, 1500]     # 第二组
        ]
        dt_ms = 1.0
        N, _, T = true_seis.shape
        time_ms = np.arange(T) * dt_ms

        def _figsave(fig, name):
            fig.savefig(os.path.join(run_dir, f"{name}.png"), dpi=400, bbox_inches='tight')
            plt.close(fig)

        # === 每组画图 ===
        for gi, traces in enumerate(trace_groups, start=1):
            fig, axes = plt.subplots(len(traces), 1, figsize=(11, 7), sharex=True)
            if len(traces) == 1: axes = [axes]
            for ax, idx in zip(axes, traces):
                idx = max(0, min(N-1, idx))  # 防止越界
                gt = true_seis[idx, 0]
                pm = pred_seis[idx, 0]

                ax.plot(time_ms, gt, color="k", linewidth=1.4, label="True")
                ax.plot(time_ms, pm, color="r", linewidth=1.2, label="Prediction")
                ax.set_ylabel("Amplitude"); ax.set_title(f"Trace {idx} — Seismic")
                ax.grid(alpha=0.25)
            axes[-1].set_xlabel("Time (ms)")
            handles, labels = axes[0].get_legend_handles_labels()
            axes[0].legend(handles, labels, loc="upper right")
            plt.tight_layout()
            _figsave(fig, f"seismic_traces_group{gi}")

        print(f"✅ All results saved in {run_dir}")


if __name__ == '__main__':
    # arguments and parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-width', type=int, default=5, help="Number of seismic traces in expanding data to be used for training. It must be odd")
    parser.add_argument('-num_train_wells', type=int, default=20, help="Number of AI traces from the model to be used for training")
    parser.add_argument('-max_epoch', type=int, default=1000, help="maximum number of training epochs")
    parser.add_argument('-batch_size', type=int, default=50, help="Batch size for training")
    parser.add_argument('-alpha', type=float, default=1, help="weight of property loss term")
    parser.add_argument('-beta', type=float, default=0.2, help="weight of seismic loss term")
    parser.add_argument('-index', type=int, default=480, help="plot index of the AI")
    parser.add_argument('-test_checkpoint', type=str, action="store", default=None, help="path to model to test on. When this flag is used, no training is performed")
    parser.add_argument('-session_name', type=str, action="store", default=datetime.now().strftime('%b%d_%H%M%S'), help="name of the session to be ised in saving the model")
    parser.add_argument('-nonlinearity', action="store", type=str, default="tanh", help="Type of nonlinearity for the CNN [tanh, relu]", choices=["tanh","relu"])
    parser.add_argument('-resolution_ratio', type=int, default=4, action="store",help="resolution mismtach between seismic and AI")
    args = parser.parse_args()

    # use the following command to modify the test_checkpoint:
    # parser.set_defaults(test_checkpoint="Nov19_205735")
    # args = parser.parse_args()
    # args.test_checkpoint = "Dec15_165634"
    if args.test_checkpoint is not None:
        test(args)
    else:
        train(args)
        test(args)