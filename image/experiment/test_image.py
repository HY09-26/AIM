"""Unified image MoRF/LeRF test — Brain Tumor MRI, ImageNet, Oxford-IIIT Pet.

Run from the image/ directory:
    python experiment/test_image.py --dataset brain_mri  --model resnet_50        --expl_method gradcam --mask_type pgd --mode morf
    python experiment/test_image.py --dataset imagenet   --model efficientnet_b0  --expl_method smoothgradcampp --mask_type road
    python experiment/test_image.py --dataset oxford_pet --model repvgg_b0        --expl_method gradcampp --mask_type pgd
"""
import os
import sys
import argparse
import pickle

import numpy as np
import torch
import torch.nn as nn
import torchvision
import timm

from torch.utils.data import DataLoader, TensorDataset, Subset
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from experiment_utils.utils import pgd_attack, pgd_attack_brainmri, road


# ── Per-dataset defaults ──────────────────────────────────────────────────────
_DATASET_INFO = {
    'brain_mri':  dict(num_classes=4,    default_epsilon=2/255),
    'imagenet':   dict(num_classes=1000, default_epsilon=8/255),
    'oxford_pet': dict(num_classes=37,   default_epsilon=2/255),
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_loader_from_np(X_np, y_np, batch_size):
    ds = TensorDataset(torch.tensor(X_np, dtype=torch.float32),
                       torch.tensor(y_np,  dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


# ── MoRF / LeRF ───────────────────────────────────────────────────────────────
def run_morf_lerf_image(
    model,
    device,
    X_np,           # (N, C, H, W)
    y_np,           # (N,)
    saliency_np,    # (N, C, H, W) or (N, 1, H, W)
    mask_np=None,   # (N, C, H, W) or None
    n_steps=20,
    batch_size=64,
    mask_type='zero',   # 'zero' | 'pgd' | 'road'
    mode='morf',        # 'morf' | 'lerf'
):
    model.eval()
    N, C, H, W = X_np.shape
    HW = H * W

    X_flat   = X_np.reshape(N, C, HW).copy()
    mask_flat = mask_np.reshape(N, C, HW) if mask_np is not None else None

    sal = np.abs(saliency_np).sum(axis=1).reshape(N, HW)
    order = np.argsort(-sal if mode == 'morf' else sal, axis=1)

    pixels_per_step = max(1, HW // n_steps)

    @torch.no_grad()
    def eval_step(X_flat_step):
        loader = DataLoader(
            TensorDataset(
                torch.tensor(X_flat_step.reshape(N, C, H, W), dtype=torch.float32),
                torch.tensor(y_np, dtype=torch.long),
            ),
            batch_size=batch_size, shuffle=False,
        )
        correct, total = 0, 0
        for xb, yb in loader:
            pred = model(xb.to(device)).argmax(dim=1)
            correct += (pred == yb.to(device)).sum().item()
            total   += yb.numel()
        return correct / max(total, 1)

    hist   = np.zeros(n_steps + 1, dtype=np.float32)
    hist[0] = eval_step(X_flat)
    print(f'[{mode.upper()}] Step 0 acc = {hist[0]:.4f}')

    X_work = X_flat.copy()
    for k in range(1, n_steps + 1):
        start = (k - 1) * pixels_per_step
        end   = min(k * pixels_per_step, HW)
        for i in range(N):
            idxs = order[i, start:end]
            if mask_type == 'zero':
                X_work[i, :, idxs] = 0.0
            elif mask_type in ('pgd', 'road'):
                if mask_flat is None:
                    raise RuntimeError(f'mask_np required for {mask_type}')
                X_work[i, :, idxs] = mask_flat[i, :, idxs]
            else:
                raise ValueError(mask_type)
        hist[k] = eval_step(X_work)
        print(f'[{mode.upper()}] Step {k}/{n_steps} acc = {hist[k]:.4f}')

    return hist


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image MoRF/LeRF evaluation')
    parser.add_argument('--dataset',     required=True,
                        choices=['brain_mri', 'imagenet', 'oxford_pet'])
    parser.add_argument('--model',       required=True,
                        choices=['resnet_50', 'efficientnet_b0', 'repvgg_b0'])
    parser.add_argument('--expl_method', required=True)
    parser.add_argument('--mask_type',   default='pgd',
                        choices=['zero', 'pgd', 'road'])
    parser.add_argument('--mode',        default='morf',
                        choices=['morf', 'lerf'])
    parser.add_argument('--n_steps',     type=int, default=20)
    parser.add_argument('--batch_size',  type=int, default=32)
    parser.add_argument('--epsilon',     type=float, default=None,
                        help='PGD epsilon; defaults per dataset')
    parser.add_argument('--pgd_steps',   type=int, default=10)
    parser.add_argument('--road_noise',  type=float, default=0.2,
                        help='ROAD noise std (oxford_pet only)')
    # Dataset-specific paths (sensible defaults)
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Checkpoint path (brain_mri / oxford_pet)')
    parser.add_argument('--expl_dir', type=str, default=None,
                        help='Saliency .npy directory; defaults to expl_image_<dataset>/<model>')
    parser.add_argument('--subset_idx', type=str, default=None,
                        help='ImageNet: .npy index file for fixed 500 subset')
    parser.add_argument('--fixed_idx_path', type=str, default=None,
                        help='Oxford-IIIT Pet: .npy fixed test indices')
    args = parser.parse_args()

    info        = _DATASET_INFO[args.dataset]
    num_classes = info['num_classes']
    epsilon     = args.epsilon if args.epsilon is not None else info['default_epsilon']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # ── Transforms ────────────────────────────────────────────────────────────
    from torchvision import transforms, datasets
    tf_list = []
    if args.dataset == 'brain_mri':
        tf_list.append(transforms.Grayscale(num_output_channels=3))
    if args.dataset == 'imagenet':
        tf_list += [transforms.Resize(256), transforms.CenterCrop(224)]
    else:
        tf_list.append(transforms.Resize((224, 224)))
    tf_list += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    tf = transforms.Compose(tf_list)

    # ── Dataset ───────────────────────────────────────────────────────────────
    if args.dataset == 'brain_mri':
        test_set = datasets.ImageFolder(
            root=os.path.join(PROJECT_ROOT, 'data', 'Brain_MRI_Tumor', 'Testing'),
            transform=tf,
        )
        test_loader = DataLoader(test_set, batch_size=args.batch_size,
                                 shuffle=False, num_workers=4)

    elif args.dataset == 'imagenet':
        val_set = datasets.ImageFolder(
            os.path.join(PROJECT_ROOT, 'data', 'ImageNet', 'val'), transform=tf
        )
        idx_path = args.subset_idx or os.path.join(
            PROJECT_ROOT, 'data', 'ImageNet', 'imagenet_val_500.npy'
        )
        fixed_idx   = np.load(idx_path).astype(int)
        test_subset = Subset(val_set, fixed_idx)
        test_loader = DataLoader(test_subset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=4)

    elif args.dataset == 'oxford_pet':
        from experiment_utils.pets_dataset import OxfordPetsDataset
        full_test = OxfordPetsDataset(
            root=os.path.join(PROJECT_ROOT, 'data', 'Oxford_Pet'),
            split='test', transform=tf,
        )
        idx_path  = args.fixed_idx_path or os.path.join(
            PROJECT_ROOT, 'experiment_utils', 'checkpoints', 'fixed_test_indices_500.npy'
        )
        fixed_idx   = np.load(idx_path).astype(int)
        test_subset = Subset(full_test, fixed_idx)
        test_loader = DataLoader(test_subset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=4)

    Xs, ys = [], []
    for x, y in tqdm(test_loader, desc='Loading test data'):
        Xs.append(x.numpy())
        ys.append(y.numpy())
    X_test = np.concatenate(Xs)
    y_test = np.concatenate(ys)
    print(f'Test samples: {len(X_test)}')

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.dataset == 'imagenet':
        # Pretrained weights
        if args.model == 'resnet_50':
            model = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        elif args.model == 'efficientnet_b0':
            model = torchvision.models.efficientnet_b0(
                weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        elif args.model == 'repvgg_b0':
            model = timm.create_model('repvgg_b0', pretrained=True, num_classes=1000)
    else:
        # Load from checkpoint
        if args.model == 'resnet_50':
            model = torchvision.models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif args.model == 'efficientnet_b0':
            model = torchvision.models.efficientnet_b0(weights=None)
            in_dim = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_dim, num_classes)
        elif args.model == 'repvgg_b0':
            model = timm.create_model('repvgg_b0', pretrained=False, num_classes=num_classes)

        ckpt_path = args.ckpt or os.path.join(
            PROJECT_ROOT, 'experiment_utils', 'checkpoints',
            args.dataset, args.model, 'best_model.pth',
        )
        print('Loading checkpoint:', ckpt_path)
        ckpt       = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        model.load_state_dict(state_dict)

    model = model.to(device).eval()

    # ── Saliency ──────────────────────────────────────────────────────────────
    expl_dir = args.expl_dir or os.path.join(
        PROJECT_ROOT, f'expl_image_{args.dataset}', args.model
    )
    saliency = np.load(os.path.join(expl_dir, f'{args.expl_method}.npy'))
    saliency = saliency[:len(X_test)]
    print('Saliency shape:', saliency.shape)

    # ── Build mask (PGD / ROAD / zero) ────────────────────────────────────────
    mask_np = None

    if args.mask_type == 'pgd':
        pgd_fn    = pgd_attack_brainmri if args.dataset == 'brain_mri' else pgd_attack
        cache_dir = os.path.join(PROJECT_ROOT, 'experiment', 'adv_cache',
                                 args.dataset, args.model)
        os.makedirs(cache_dir, exist_ok=True)

        if args.dataset == 'oxford_pet':
            subset_tag = os.path.splitext(os.path.basename(idx_path))[0]
            eps_tag    = f'{epsilon:.6f}'.replace('.', 'p')
            cache_path = os.path.join(
                cache_dir, f'{subset_tag}_pgd_eps{eps_tag}_steps{args.pgd_steps}.npy'
            )
        else:
            cache_path = os.path.join(
                cache_dir, f'pgd_eps{epsilon}_steps{args.pgd_steps}.npy'
            )

        if os.path.exists(cache_path):
            print('[PGD] Loading cached adversarial examples')
            mask_np = np.load(cache_path)
        else:
            print('[PGD] Running PGD (cached on first run)...')
            loader_pgd = build_loader_from_np(X_test, y_test, args.batch_size)
            adv = pgd_fn(model, device, loader_pgd, nn.CrossEntropyLoss(),
                         epsilon=epsilon, steps=args.pgd_steps)
            mask_np = adv.detach().cpu().numpy()
            np.save(cache_path, mask_np)
            print('[PGD] Saved to:', cache_path)
            del adv
            torch.cuda.empty_cache()

    elif args.mask_type == 'road':
        cache_dir = os.path.join(PROJECT_ROOT, 'experiment', 'road_cache', args.dataset)
        os.makedirs(cache_dir, exist_ok=True)

        if args.dataset == 'oxford_pet':
            subset_tag  = os.path.splitext(os.path.basename(idx_path))[0]
            noise_tag   = f'{args.road_noise:.3f}'.replace('.', 'p')
            cache_path  = os.path.join(cache_dir, f'{subset_tag}_road_noise{noise_tag}.npy')
            road_loader = test_loader
        else:
            cache_path  = os.path.join(cache_dir, 'road_noise0.2.npy')
            road_loader = build_loader_from_np(X_test, y_test, args.batch_size)

        if os.path.exists(cache_path):
            print('[ROAD] Loading cached ROAD images')
            mask_np = np.load(cache_path)
        else:
            print('[ROAD] Generating ROAD images (cached on first run)...')
            mask_np = road(road_loader, noise_std=args.road_noise)
            np.save(cache_path, mask_np)
            print('[ROAD] Saved to:', cache_path)

    # ── Run MoRF / LeRF ───────────────────────────────────────────────────────
    hist = run_morf_lerf_image(
        model=model, device=device,
        X_np=X_test, y_np=y_test, saliency_np=saliency,
        mask_np=mask_np, n_steps=args.n_steps, batch_size=args.batch_size,
        mask_type=args.mask_type, mode=args.mode,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = os.path.join(PROJECT_ROOT, 'morf_lerf_image',
                           args.model, args.dataset, args.mask_type)
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f'{args.expl_method}_{args.mode}.pkl')
    with open(save_path, 'wb') as fh:
        pickle.dump(hist, fh)
    print('Saved results to:', save_path)
