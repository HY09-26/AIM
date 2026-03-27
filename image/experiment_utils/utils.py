import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


# ============================================================
# Basic metrics
# ============================================================
@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    Multi-class accuracy
    logits: (B, C)
    y: (B,)
    """
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


# ============================================================
# Train / Eval (classification)
# ============================================================
def train_one_epoch(
    model: nn.Module,
    device,
    loader,
    loss_fn,
    optimizer,
):
    model.train()
    losses, accs = [], []

    for i, (x, y) in enumerate(
        tqdm(loader, desc="Train", leave=False)
    ):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accs.append(accuracy_from_logits(logits.detach(), y))

    return float(np.mean(losses)), float(np.mean(accs))


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    device,
    loader,
    loss_fn,
    return_logits: bool = False,
):
    model.eval()
    losses, accs = [], []
    all_logits = [] if return_logits else None

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = loss_fn(logits, y)

        losses.append(loss.item())
        accs.append(accuracy_from_logits(logits, y))

        if return_logits:
            all_logits.append(logits.detach().cpu())

    if return_logits:
        return (
            float(np.mean(losses)),
            float(np.mean(accs)),
            torch.cat(all_logits, dim=0).numpy(),
        )

    return float(np.mean(losses)), float(np.mean(accs))


# ============================================================
# Binary classification (AUC) – for ChestX-ray8
# ============================================================
def train_one_epoch_auc(
    model,
    device,
    loader,
    loss_fn,
    optimizer,
):
    model.train()
    losses = []
    y_true, y_prob = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        prob1 = torch.softmax(logits.detach(), dim=1)[:, 1]
        y_prob.append(prob1.cpu().numpy())
        y_true.append(y.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    auc = roc_auc_score(y_true, y_prob)

    return float(np.mean(losses)), float(auc)


@torch.no_grad()
@torch.no_grad()
def eval_one_epoch_auc(
    model,
    device,
    loader,
    loss_fn,
    return_logits: bool = False,
):
    """
    NOTE:
    - 名字保留 eval_one_epoch_auc（避免動到外部 training pipeline）
    - 實際 metric 改為 Accuracy（argmax）
    """
    model.eval()
    losses = []
    correct, total = 0, 0
    all_logits = [] if return_logits else None

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = loss_fn(logits, y)
        losses.append(loss.item())

        # -------- Accuracy --------
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()

        if return_logits:
            all_logits.append(logits.detach().cpu())

    acc = correct / max(total, 1)

    if return_logits:
        return float(np.mean(losses)), float(acc), torch.cat(all_logits).numpy()

    return float(np.mean(losses)), float(acc)


# ============================================================
# Unified fit loop (SAVE BEST)
# ============================================================
def fit_classifier(
    model,
    device,
    train_loader,
    eval_loader,
    loss_fn,
    optimizer,
    epochs,
    ckpt_dir,
    ckpt_best="best.pth",
    ckpt_last="last.pth",
    task="multiclass",   # "multiclass" or "binary"
):
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- sanity prints  ---
    print(f"[fit_classifier] task={task}, epochs={epochs}")
    print(f"[fit_classifier] train_batches={len(train_loader)}, eval_batches={len(eval_loader)}")
    print(f"[fit_classifier] ckpt_dir={ckpt_dir}")
    print(f"[fit_classifier] ckpt_best={ckpt_best}, ckpt_last={ckpt_last}")

    best_metric = -1e9
    best_epoch = -1
    best_path = os.path.join(ckpt_dir, ckpt_best)

    for ep in range(1, epochs + 1):
        if task == "binary":
            tr_loss, tr_metric = train_one_epoch_auc(
                model, device, train_loader, loss_fn, optimizer
            )
            ev_loss, ev_metric = eval_one_epoch_auc(
                model, device, eval_loader, loss_fn
            )
            metric_name = "AUC"
        else:
            tr_loss, tr_metric = train_one_epoch(
                model, device, train_loader, loss_fn, optimizer
            )
            ev_loss, ev_metric = eval_one_epoch(
                model, device, eval_loader, loss_fn
            )
            metric_name = "Acc"

        print(
            f"Epoch {ep:02d}/{epochs} | "
            f"Train Loss {tr_loss:.4f} {metric_name} {tr_metric:.4f} | "
            f"Eval  Loss {ev_loss:.4f} {metric_name} {ev_metric:.4f}"
        )

        if ev_metric > best_metric:
            best_metric = ev_metric
            best_epoch = ep
            torch.save(model.state_dict(), best_path)

    last_path = os.path.join(ckpt_dir, ckpt_last)
    torch.save(model.state_dict(), last_path)
    print(f"Saved LAST -> {last_path}")

    print("\n================ Summary ================")
    print(f"Best epoch : {best_epoch}")
    print(f"Best metric: {best_metric:.4f}")
    print(f"Best model : {best_path}")
    print("========================================")

    return {
        "best_epoch": best_epoch,
        "best_metric": float(best_metric),
        "best_path": best_path,
        "last_path": last_path,
    }


# ============================================================
# PGD (image)
# ============================================================
def pgd_attack(
    model,
    device,
    loader,
    loss_fn,
    epsilon=2/255,
    steps=10,
    random_start=True,
    clip_min=0.0,
    clip_max=1.0,
):
    model.eval()
    adv_all = []
    step = epsilon / steps

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        x0 = x.detach()

        if random_start:
            x = x + (torch.rand_like(x) * 2 - 1) * step
            x = torch.clamp(x, clip_min, clip_max)

        for _ in range(steps):
            x.requires_grad_(True)
            logits = model(x)
            loss = loss_fn(logits, y)
            grad = torch.autograd.grad(loss, x)[0]

            x = x.detach() + step * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x0 - epsilon), x0 + epsilon)
            x = torch.clamp(x, clip_min, clip_max)

        adv_all.append(x.detach().cpu())

    return torch.cat(adv_all, dim=0)




def pgd_attack_brainmri(
    model,
    device,
    loader,
    loss_fn,
    epsilon=2/255,
    alpha=1/255,
    steps=10,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    """
    PGD for Brain MRI (raw-space constrained, normalized-space forward)

    Returns
    -------
    adv_x : torch.Tensor, shape (N,3,H,W), normalized
    """

    model.eval()
    adv_all = []

    mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(std,  device=device).view(1, 3, 1, 1)

    for x_norm, y in loader:
        x_norm = x_norm.to(device)
        y = y.to(device)

        # --------------------------------------------------
        # Recover raw image in [0,1]
        # --------------------------------------------------
        x_raw = x_norm * std + mean
        x_raw = torch.clamp(x_raw, 0.0, 1.0)

        # init
        x_adv = x_raw.clone().detach()

        for _ in range(steps):
            x_adv.requires_grad_(True)

            # forward uses normalized input
            x_adv_norm = (x_adv - mean) / std
            logits = model(x_adv_norm)
            loss = loss_fn(logits, y)

            grad = torch.autograd.grad(loss, x_adv)[0]

            # PGD update in raw space
            x_adv = x_adv + alpha * grad.sign()

            # projection
            x_adv = torch.min(
                torch.max(x_adv, x_raw - epsilon),
                x_raw + epsilon
            )

            # clamp to valid MRI intensity
            x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

        # return normalized adversarial
        adv_all.append(((x_adv - mean) / std).detach().cpu())

    return torch.cat(adv_all, dim=0)



# ============================================================
# ROAD
# ============================================================
import torch.nn.functional as F


@torch.no_grad()
def road(

    loader,    # yields (x, y) where x is normalized tensor (B,3,H,W)
    noise_std=0.2,
):
    """
    Generate ONE debiased replacement image per sample.
    Return: (N, 3, H, W) numpy float32
    """
    X_all = []
    for xb, _ in tqdm(loader, desc="[ROAD] collect X"):
        X_all.append(xb.numpy())
    X_all = np.concatenate(X_all, axis=0).astype(np.float32)  # (N,3,H,W)

    X = torch.from_numpy(X_all)  # CPU tensor
    N, C, H, W = X.shape

    wd, wi = 1/6, 1/12
    Xp = F.pad(X, (1,1,1,1), mode="reflect")

    up    = Xp[:, :, 0:H,   1:W+1]
    down  = Xp[:, :, 2:H+2, 1:W+1]
    left  = Xp[:, :, 1:H+1, 0:W]
    right = Xp[:, :, 1:H+1, 2:W+2]

    ul = Xp[:, :, 0:H,   0:W]
    ur = Xp[:, :, 0:H,   2:W+2]
    dl = Xp[:, :, 2:H+2, 0:W]
    dr = Xp[:, :, 2:H+2, 2:W+2]

    interp = wd*(up+down+left+right) + wi*(ul+ur+dl+dr)
    noise = noise_std * torch.randn_like(interp)

    X_road = (interp + noise).numpy().astype(np.float32)
    return X_road
