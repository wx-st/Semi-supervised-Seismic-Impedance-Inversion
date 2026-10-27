import argparse
import os, random

# ========= Reproducibility (use your original random_seed name) =========
random_seed = 30  # 保持原名不变
os.environ["PYTHONHASHSEED"] = str(random_seed)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# =======================================================================

import torch
import numpy as np
from os.path import isdir
from models.the_proposed_method import KANInverseModel as inverse_model, KANForwardModel as forward_model
from torch.utils import data
from core.functions import *
from torch import nn, optim
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import csv
import torch.nn.functional as F
import json



print(torch.cuda.is_available())
print(torch.version.cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


@torch.no_grad()
def _batch_corr(yhat, y, eps=1e-8):
    if yhat.dim() == 3:
        yhat = yhat.squeeze(1)
    if y.dim() == 3:
        y = y.squeeze(1)
    x = yhat - yhat.mean(dim=-1, keepdim=True)
    z = y - y.mean(dim=-1, keepdim=True)
    num = (x * z).sum(dim=-1)
    den = torch.sqrt((x * x).sum(dim=-1) * (z * z).sum(dim=-1) + eps)
    corr = num / (den + eps)
    return corr.mean()


@torch.no_grad()
def _batch_r2(yhat, y, eps=1e-8):
    if yhat.dim() == 3:
        yhat = yhat.squeeze(1)
    if y.dim() == 3:
        y = y.squeeze(1)
    y_mean = y.mean(dim=-1, keepdim=True)
    sse = ((y - yhat) ** 2).sum(dim=-1)
    sst = ((y - y_mean) ** 2).sum(dim=-1) + eps
    r2 = 1.0 - sse / sst
    return r2.mean()


def set_seed_all(seed: int = 30, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


torch_generator = set_seed_all(random_seed, deterministic=True)
warnings.filterwarnings('ignore')


class MultiTraceDataset(data.Dataset):
    """
    seismic:   [N, 1, T_low]
    impedance: [N, 1, T_high]
    返回:
        x_win: [1, in_width, T_low]
        y_win: [out_width, T_high]
        center: int
    """
    def __init__(self, seismic, impedance, in_width=9, out_width=5, trace_indices=None):
        super().__init__()
        assert in_width % 2 == 1, "in_width must be odd"
        assert out_width % 2 == 1, "out_width must be odd"
        assert out_width <= in_width, "out_width must be <= in_width"

        self.seismic = seismic
        self.impedance = impedance
        self.in_width = in_width
        self.out_width = out_width

        self.N = seismic.shape[0]
        self.in_half = in_width // 2
        self.out_half = out_width // 2

        valid_centers = np.arange(self.in_half, self.N - self.in_half)

        if trace_indices is not None:
            trace_indices = np.array(trace_indices)
            valid_centers = np.array([c for c in valid_centers if c in trace_indices])

        self.centers = valid_centers

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx):
        c = self.centers[idx]

        in_l = c - self.in_half
        in_r = c + self.in_half + 1

        out_l = c - self.out_half
        out_r = c + self.out_half + 1

        x_win = self.seismic[in_l:in_r]      # [in_width,1,T_low]
        y_win = self.impedance[out_l:out_r]  # [out_width,1,T_high]

        x_win = x_win.permute(1, 0, 2)       # [1,in_width,T_low]
        y_win = y_win.squeeze(1)             # [out_width,T_high]

        return x_win, y_win, c


def get_data(args, test=False):
    try:
        data_dic = np.load("data/data.npy", allow_pickle=True).item()
        seismic_data = data_dic["seismic"]
        acoustic_impedance_data = data_dic["acoustic_impedance"]

        print(f"地震数据形状: {seismic_data.shape}")
        print(f"声波阻抗数据形状: {acoustic_impedance_data.shape}")
        print(f"resolution_ratio: {args.resolution_ratio}")
        print(f"in_width: {args.in_width}, out_width: {args.out_width}")

        seismic_mean = torch.tensor(np.mean(seismic_data, keepdims=True)).float()
        seismic_std = torch.tensor(np.std(seismic_data, keepdims=True)).float()

        acoustic_mean = torch.tensor(np.mean(acoustic_impedance_data, keepdims=True)).float()
        acoustic_std = torch.tensor(np.std(acoustic_impedance_data, keepdims=True)).float()

        seismic_data = torch.tensor(seismic_data).float().to(device)
        acoustic_impedance_data = torch.tensor(acoustic_impedance_data).float().to(device)

        seismic_mean = seismic_mean.to(device)
        seismic_std = seismic_std.to(device)
        acoustic_mean = acoustic_mean.to(device)
        acoustic_std = acoustic_std.to(device)

        seismic_normalization = Normalization(mean_val=seismic_mean, std_val=seismic_std)
        acoustic_normalization = Normalization(mean_val=acoustic_mean, std_val=acoustic_std)

        seismic_data = seismic_normalization.normalize(seismic_data)
        acoustic_impedance_data = acoustic_normalization.normalize(acoustic_impedance_data)

        num_samples = seismic_data.shape[0]
        indecies = np.arange(0, num_samples)
        train_indecies = indecies[(np.linspace(0, len(indecies)-1, args.num_train_wells)).astype(int)]

        if not test:
            train_dataset = MultiTraceDataset(
                seismic=seismic_data,
                impedance=acoustic_impedance_data,
                in_width=args.in_width,
                out_width=args.out_width,
                trace_indices=train_indecies
            )

            unlabeled_dataset = MultiTraceDataset(
                seismic=seismic_data,
                impedance=acoustic_impedance_data,
                in_width=args.in_width,
                out_width=args.out_width,
                trace_indices=None
            )

            train_loader = data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                generator=torch_generator,
                worker_init_fn=lambda wid: np.random.seed(random_seed + wid),
                num_workers=0,
                pin_memory=False
            )

            unlabeled_loader = data.DataLoader(
                unlabeled_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                generator=torch_generator,
                worker_init_fn=lambda wid: np.random.seed(random_seed + wid),
                num_workers=0,
                pin_memory=False
            )

            return train_loader, unlabeled_loader, seismic_normalization, acoustic_normalization

        else:
            test_dataset = MultiTraceDataset(
                seismic=seismic_data,
                impedance=acoustic_impedance_data,
                in_width=args.in_width,
                out_width=args.out_width,
                trace_indices=None
            )

            test_loader = data.DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                generator=torch_generator,
                worker_init_fn=lambda wid: np.random.seed(random_seed + wid),
                num_workers=0,
                pin_memory=False
            )

            return test_loader, seismic_normalization, acoustic_normalization

    except Exception as e:
        print(f"Error in get_data: {e}")
        import traceback
        traceback.print_exc()
        raise


def get_models(args):
    if args.test_checkpoint is None:
        inverse_net = inverse_model(
            in_width=args.in_width,
            out_width=args.out_width,
            nonlinearity=args.nonlinearity
        )
        forward_net = forward_model(
            resolution_ratio=args.resolution_ratio,
            nonlinearity=args.nonlinearity
        )
        optimizer = optim.Adam(
            list(inverse_net.parameters()) + list(forward_net.parameters()),
            amsgrad=True,
            lr=0.001
        )
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
    forward_net.train()
    criterion = nn.MSELoss()

    print(f"Number of batches in train_loader: {len(train_loader)}")
    if len(train_loader) == 0:
        raise ValueError("train_loader is empty! Check data loading.")

    if not isdir("checkpoints"):
        os.mkdir("checkpoints")

    log_path = os.path.join("logs", f"{args.session_name}_train_metrics.csv")
    _ensure_dir(log_path)
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "property_loss", "seismic_loss",
                "mse_y", "mse_x", "r2", "corr"
            ])

    loss_var = []
    print("Training the model")

    unlabeled_iter = iter(unlabeled_loader) if args.beta != 0 else None

    in_mid_left = (args.in_width - args.out_width) // 2
    in_mid_right = in_mid_left + args.out_width

    for epoch in range(args.max_epoch):
        batch_losses = []
        ep_prop_losses = []
        ep_seis_losses = []
        ep_mse_y = []
        ep_mse_x = []
        ep_r2 = []
        ep_corr = []

        for x_win, y_win, _ in train_loader:
            optimizer.zero_grad()

            # x_win: [B,1,in_width,T_low]
            # y_win: [B,out_width,T_high]
            y_pred = inverse_net(x_win)                  # [B,out_width,T_high]
            x_rec = forward_net(y_win)                   # [B,out_width,T_low]

            # 输入窗口中部 out_width 道作为目标地震区
            x_target = x_win[:, 0, in_mid_left:in_mid_right, :]   # [B,out_width,T_low]

            # ===== 监督损失 =====
            property_loss = criterion(y_pred, y_win) + criterion(x_rec, x_target)

            # ===== 无监督正演损失 =====
            if args.beta != 0:
                try:
                    x_u, _, _ = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    x_u, _, _ = next(unlabeled_iter)

                x_u_target = x_u[:, 0, in_mid_left:in_mid_right, :]   # [B,out_width,T_low]
                y_u_pred = inverse_net(x_u)                           # [B,out_width,T_high]
                x_u_rec = forward_net(y_u_pred)                       # [B,out_width,T_low]
                seismic_loss = criterion(x_u_rec, x_u_target)
            else:
                seismic_loss = torch.tensor(0.0, device=y_win.device)

            # ===== 总损失 =====
            loss = (
                args.alpha * property_loss +
                args.beta * seismic_loss
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(inverse_net.parameters()) + list(forward_net.parameters()),
                max_norm=10.0
            )
            optimizer.step()

            with torch.no_grad():
                mse_y = F.mse_loss(y_pred, y_win)
                mse_x = F.mse_loss(x_rec, x_target)

                yp = y_pred.reshape(-1, 1, y_pred.shape[-1])
                yt = y_win.reshape(-1, 1, y_win.shape[-1])

                r2 = _batch_r2(yp, yt)
                corr = _batch_corr(yp, yt)

                ep_prop_losses.append(property_loss.detach().item())
                ep_seis_losses.append(seismic_loss.detach().item())
                ep_mse_y.append(mse_y.detach().item())
                ep_mse_x.append(mse_x.detach().item())
                ep_r2.append(r2.detach().item())
                ep_corr.append(corr.detach().item())

            batch_losses.append(loss.detach().item())

        train_loss = float(np.mean(batch_losses))
        loss_var.append(train_loss)

        mean_prop = float(np.mean(ep_prop_losses))
        mean_seis = float(np.mean(ep_seis_losses))
        mean_mse_y = float(np.mean(ep_mse_y))
        mean_mse_x = float(np.mean(ep_mse_x))
        mean_r2 = float(np.mean(ep_r2))
        mean_corr = float(np.mean(ep_corr))

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_loss,
                mean_prop,
                mean_seis,
                mean_mse_y,
                mean_mse_x,
                mean_r2,
                mean_corr
            ])

        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch+1}: "
                f"loss={train_loss:.4f}, "
                f"prop_loss={mean_prop:.4f}, "
                f"seis_loss={mean_seis:.4f}"
            )

    loss_var = torch.tensor(loss_var).cpu().numpy()
    run_dir = os.path.join("results", args.session_name)
    os.makedirs(run_dir, exist_ok=True)
    np.save(os.path.join(run_dir, "train_loss.npy"), loss_var)

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
    test_loader, seismic_normalization, acoustic_normalization = get_data(args, test=True)
    if args.test_checkpoint is None:
        args.test_checkpoint = args.session_name
    inverse_net, forward_net, _ = get_models(args)

    run_dir = os.path.join("results", args.session_name)
    os.makedirs(run_dir, exist_ok=True)

    data_dic = np.load("data/data.npy", allow_pickle=True).item()
    seismic_raw = data_dic["seismic"]               # [N,1,T_low]
    impedance_raw = data_dic["acoustic_impedance"]  # [N,1,T_high]

    N = seismic_raw.shape[0]
    T_low = seismic_raw.shape[-1]
    T_high = impedance_raw.shape[-1]

    # ========= 加权融合累加器 =========
    pred_sum = torch.zeros((N, T_high), device=device)
    pred_weight = torch.zeros((N, T_high), device=device)

    pred_seis_sum = torch.zeros((N, T_low), device=device)
    pred_seis_weight = torch.zeros((N, T_low), device=device)

    true_impedance = torch.tensor(impedance_raw, dtype=torch.float32, device=device).squeeze(1)
    true_seismic = torch.tensor(seismic_raw, dtype=torch.float32, device=device).squeeze(1)

    inverse_net.eval()
    forward_net.eval()
    print("\nTesting the model with weighted overlap fusion\n")

    test_loss = []
    out_half = args.out_width // 2

    # ========= 构造横向融合权重 =========
    center = out_half
    w = []
    for i in range(args.out_width):
        w.append(out_half + 1 - abs(i - center))
    trace_weights = torch.tensor(w, dtype=torch.float32, device=device)


    # 归一化不是必须，但便于数值稳定和理解
    trace_weights = trace_weights / trace_weights.sum()

    print(f"Fusion weight type: {getattr(args, 'fusion_weight_type', 'triangular')}")
    print(f"Trace fusion weights: {trace_weights.detach().cpu().numpy()}")

    with torch.no_grad():
        for x_win, y_win, centers in test_loader:
            # x_win: [B,1,in_width,T_low]
            # y_win: [B,out_width,T_high]
            # centers: [B]
            y_pred = inverse_net(x_win)      # [B,out_width,T_high]
            x_rec = forward_net(y_pred)      # [B,out_width,T_low]

            loss = F.mse_loss(y_pred, y_win)
            test_loss.append(loss.item())

            B = x_win.shape[0]
            for b in range(B):
                c = int(centers[b].item())
                out_l = c - out_half
                out_r = c + out_half + 1

                # ===== 对每个输出道施加不同融合权重 =====

                w_ai = trace_weights.unsqueeze(-1)    # [out_width,1]
                w_seis = trace_weights.unsqueeze(-1)  # [out_width,1]

                pred_sum[out_l:out_r] += y_pred[b] * w_ai
                pred_weight[out_l:out_r] += w_ai

                pred_seis_sum[out_l:out_r] += x_rec[b] * w_seis
                pred_seis_weight[out_l:out_r] += w_seis

        # ========= 加权平均融合 =========
        pred_impedance_norm = pred_sum / (pred_weight + 1e-8)
        pred_seis_norm = pred_seis_sum / (pred_seis_weight + 1e-8)

        # 真实值归一化，用于指标统计
        seismic_norm = seismic_normalization.normalize(true_seismic.unsqueeze(1)).squeeze(1)
        impedance_norm = acoustic_normalization.normalize(true_impedance.unsqueeze(1)).squeeze(1)

        valid_mask = pred_weight[:, 0] > 0
        pred_valid = pred_impedance_norm[valid_mask]
        true_valid = impedance_norm[valid_mask]

        property_corr = _batch_corr(pred_valid.unsqueeze(1), true_valid.unsqueeze(1)).item()
        property_r2 = _batch_r2(pred_valid.unsqueeze(1), true_valid.unsqueeze(1)).item()
        loss_scalar = float(np.mean(test_loss))

        print(f"loss: {loss_scalar:.4f}\nCorrelation: {property_corr:.4f}\nr2 Coeff.: {property_r2:.4f}")

        # ========= 反归一化 =========
        predicted_impedance = acoustic_normalization.unnormalize(pred_impedance_norm.unsqueeze(1))
        true_impedance_full = acoustic_normalization.unnormalize(impedance_norm.unsqueeze(1))
        pred_seis = seismic_normalization.unnormalize(pred_seis_norm.unsqueeze(1))
        true_seis = seismic_normalization.unnormalize(seismic_norm.unsqueeze(1))

        predicted_impedance = predicted_impedance.cpu().numpy()
        true_impedance_full = true_impedance_full.cpu().numpy()
        true_seis = true_seis.cpu().numpy()
        pred_seis = pred_seis.cpu().numpy()

        # ========= 2D 剖面图 =========
        true_sec = true_seis[:, 0, :]
        pred_sec = pred_seis[:, 0, :]
        err_sec = pred_sec - true_sec

        def _save_section(img, title, fname, vmin=None, vmax=None, cmap='viridis'):
            fig = plt.figure(figsize=(12, 6))
            plt.imshow(
                img.T,
                aspect='auto',
                origin='upper',
                vmin=vmin,
                vmax=vmax,
                cmap=cmap
            )
            plt.title(title)
            plt.xlabel("Trace Index")
            plt.ylabel("Time Sample")
            plt.colorbar()
            fig.savefig(os.path.join(run_dir, fname), dpi=300, bbox_inches='tight')
            plt.close(fig)

        seis_vmin = np.min(true_sec)
        seis_vmax = np.max(true_sec)
        _save_section(true_sec, "True Seismic Section", "section_true_seis.png",
                      vmin=seis_vmin, vmax=seis_vmax, cmap='gray')
        _save_section(pred_sec, "Predicted Seismic Section", "section_pred_seis.png",
                      vmin=seis_vmin, vmax=seis_vmax, cmap='gray')
        _save_section(err_sec, "Error Seismic Section (Pred-True)", "section_err_seis.png",
                      cmap='gray')

        true_ai = true_impedance_full[:, 0, :]
        pred_ai = predicted_impedance[:, 0, :]
        err_ai = np.abs(pred_ai - true_ai)

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

        _save_section(
            err_ai,
            "Error AI Section (Pred-True)",
            "section_err_ai.png",
            vmin=0.0,
            vmax=7000,
            cmap='binary'
        )

        # ========= 指标 =========
        psnr = PSNR(true_impedance_full[:, 0], predicted_impedance[:, 0])

        print(f"PSNR: {psnr:.4f}\n")

        with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
            json.dump({
                "loss": float(loss_scalar),
                "corr": float(property_corr),
                "r2": float(property_r2),
                "psnr": float(psnr),
                "session_name": args.session_name,
                "fusion_weight_type": getattr(args, "fusion_weight_type", "triangular")
            }, f, indent=2)

        np.save(os.path.join(run_dir, "true_impedance.npy"), true_impedance_full)
        np.save(os.path.join(run_dir, "pred_impedance.npy"), predicted_impedance)
        np.save(os.path.join(run_dir, "true_seis.npy"), true_seis)
        np.save(os.path.join(run_dir, "pred_seis.npy"), pred_seis)

        # 保存融合权重，方便记录实验
        np.save(os.path.join(run_dir, "trace_fusion_weights.npy"), trace_weights.detach().cpu().numpy())

        # ========= 固定剖面选择 =========
        trace_groups = [
            [625, 1250, 1875],
            [300, 900, 1500]
        ]
        dt_ms = 1.0
        N_plot, _, T_plot = true_seis.shape
        time_ms = np.arange(T_plot) * dt_ms

        def _figsave(fig, name):
            fig.savefig(os.path.join(run_dir, f"{name}.png"), dpi=400, bbox_inches='tight')
            plt.close(fig)

        for gi, traces in enumerate(trace_groups, start=1):
            fig, axes = plt.subplots(len(traces), 1, figsize=(11, 7), sharex=True)
            if len(traces) == 1:
                axes = [axes]
            for ax, idx in zip(axes, traces):
                idx = max(0, min(N_plot - 1, idx))
                gt = true_seis[idx, 0]
                pm = pred_seis[idx, 0]

                ax.plot(time_ms, gt, color="k", linewidth=1.4, label="True")
                ax.plot(time_ms, pm, color="r", linewidth=1.2, label="Prediction")
                ax.set_ylabel("Amplitude")
                ax.set_title(f"Trace {idx} — Seismic")
                ax.grid(alpha=0.25)
            axes[-1].set_xlabel("Time (ms)")
            handles, labels = axes[0].get_legend_handles_labels()
            axes[0].legend(handles, labels, loc="upper right")
            plt.tight_layout()
            _figsave(fig, f"seismic_traces_group{gi}")

        print(f"✅ All results saved in {run_dir}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-num_train_wells', type=int, default=20, help="Number of AI traces from the model to be used for training")
    parser.add_argument('-max_epoch', type=int, default=1000, help="maximum number of training epochs")
    parser.add_argument('-batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('-alpha', type=float, default=1.0, help="weight of property loss term")
    parser.add_argument('-beta', type=float, default=0.2, help="weight of seismic loss term")
    parser.add_argument('-index', type=int, default=480, help="plot index of the AI")
    parser.add_argument('-test_checkpoint', type=str, action="store", default=None, help="path to model to test on. When this flag is used, no training is performed")
    parser.add_argument('-session_name', type=str, action="store", default=datetime.now().strftime('%b%d_%H%M%S'), help="name of the session to be ised in saving the model")
    parser.add_argument('-nonlinearity', action="store", type=str, default="tanh", help="Type of nonlinearity for the CNN [tanh, relu]", choices=["tanh","relu"])
    parser.add_argument('-resolution_ratio', type=int, default=4, action="store", help="resolution mismtach between seismic and AI")

    parser.add_argument('--T', type=int, default=470, help='时间/深度维度长度，默认470')
    parser.add_argument('--in_width', type=int, default=9, help='输入地震窗口宽度，必须为奇数')
    parser.add_argument('--out_width', type=int, default=7, help='输出阻抗窗口宽度，必须为奇数，且 <= in_width')

    args = parser.parse_args()

    assert args.in_width % 2 == 1, "in_width must be odd"
    assert args.out_width % 2 == 1, "out_width must be odd"
    assert args.out_width <= args.in_width, "out_width must be <= in_width"

    if args.test_checkpoint is not None:
        test(args)
    else:
        train(args)
        test(args)