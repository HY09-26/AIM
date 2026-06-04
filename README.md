# AIM: Adversarial Information Masking for Faithfulness Evaluation of Saliency Maps


Post-hoc saliency methods are widely used to interpret deep neural networks, but their faithfulness is difficult to evaluate reliably. Existing evaluations mask features according to saliency-induced orderings and measure performance degradation, but this degradation can be confounded by the masking operator: zero masking may create out-of-distribution artifacts, while interpolation-based masking may preserve residual predictive information.

We propose **AIM (Adversarial Information Masking)**, a saliency-guided adversarial feature replacement framework. AIM replaces selected features with values from a PGD-generated adversarial counterpart and compares degradation under complementary MoRF/LeRF masking orders. Experiments across image, audio, and EEG tasks suggest that AIM reduces masking-induced bias compared with zero and interpolation-based masking.

---

## Repository Structure

```
AIM/
├── image/                          # Image modality
│   ├── experiment/
│   │   ├── test_image.py           # MoRF/LeRF on Brain Tumor MRI / ImageNet / Oxford-IIIT Pet
│   │   ├── area.py                 # Compute AOC / ABC / AUC
│   │   ├── spearman.py             # MoRF–LeRF Spearman correlation
│   │   └── plot_morf_lerf.py       # Plot MoRF/LeRF curves
│   └── experiment_utils/
│       ├── model.py                # ResNet-50, EfficientNet-B0, RepVGG-B0
│       ├── utils.py                # PGD attack, ROAD masking, helpers
│       ├── image_loader.py         # Dataset loading
│
├── audio/                          # Audio modality
│   ├── experiment/
│   │   ├── test_waveform.py        # MoRF/LeRF on 1-D waveforms (rank-based)
│   │   ├── test_waveform_interval.py  # MoRF/LeRF on 1-D waveforms (interval-based)
│   │   ├── test_spectrogram.py     # MoRF/LeRF on 2-D spectrogram / log-mel
│   │   ├── area.py                 # Compute AOC / ABC / AUC
│   │   ├── spearman.py             # MoRF–LeRF Spearman correlation
│   │   └── plot_morf_lerf.py       # Plot MoRF/LeRF curves
│   └── experiment_utils/
│       ├── model/                  # AudioNet, Res1dNet31, AlexNet_Audio, CNN14, ...
│       ├── utils.py                # PGD attack, ROAD/MFBB masking, loaders
│       ├── mfbb.py                 # Multipoint Fractional Brownian Bridge
│       ├── train_audio.py          # Unified training (AudioMNIST / ESC-50 / MSoS)
│
└── eeg/                            # EEG modality
    ├── experiment/
    │   ├── chzero_test.py          # Channel masking — zero replacement
    │   ├── chae_test.py            # Channel masking — adversarial (AIM)
    │   ├── chroad_test.py          # Channel masking — spatial imputation (mdROAD)
    │   ├── fqzero_test.py          # Frequency masking — zero replacement
    │   ├── fqae_test.py            # Frequency masking — adversarial (AIM)
    │   ├── fqroad_test.py          # Frequency masking — spectral imputation (mdROAD)
    │   ├── tszero_test.py          # Time-segment masking — zero replacement
    │   ├── tsae_test.py            # Time-segment masking — adversarial (AIM)
    │   └── tsroad_test.py          # Time-segment masking — MFBB imputation (mdROAD)
    ├── experiment_utils/
    │   ├── eeg_config.py           # Per-dataset configuration (paths, kwerg, sub_list, …)
    │   ├── masking_utils.py        # find_neighbors, find_crop, NoisySpatialImputer, electrode topology
    │   ├── model.py                # EEGNet, InterpretableCNN, SCCNet (+ SSVEP variants)
    │   ├── utils.py                # Training/evaluation loops, data loaders
    │   ├── expl_gen.py             # Saliency map generation
    │   ├── mfbb.py                 # Multipoint Fractional Brownian Bridge
    │   └── model_train.py          # Training pipeline
    └── result_process/             # Result aggregation and visualization
        ├── result_areas.py         # AOC / ABC / AUC computation
        ├── result_spears.py        # Spearman consistency analysis
        ├── result_grid.py          # Grid-format result tables
        └── plot.py                 # Plotting utilities
```

---



## Datasets

| Modality | Dataset | Classes | Task |
|----------|---------|---------|------|
| **Image** | [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) | 37 | Fine-grained pet breed classification |
| | [ImageNet](https://www.image-net.org/) | 1000 | Large-scale object recognition |
| | [Brain Tumor MRI](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) | 4 | Medical image classification |
| **Audio** | [AudioMNIST](https://github.com/soerenab/AudioMNIST) | 10 | Spoken digit recognition |
| | [ESC-50](https://github.com/karolpiczak/ESC-50) | 50 | Environmental sound classification |
| | [MSoS](https://zenodo.org/record/3553021) | 5 | Complex acoustic scene classification |
| **EEG** | [SMR / BCI IV 2a](https://www.bbci.de/competition/iv/) | 4 | Motor imagery (9 subjects) |
| | [ERN / BCI Challenge](https://www.kaggle.com/c/inria-bci-challenge) | 2 | Error-related negativity (16 subjects) |
| | [SSVEP / MAMEM](https://www.mamem.eu/results/datasets/) | 5 | Steady-state visual evoked potential (11 subjects) |

ImageNet and Oxford-IIIT Pet use a fixed random subset of 500 samples. ESC-50 uses 5-fold cross-validation. EEG models are trained subject-specifically across 5 random seeds.

---

## Models

### Image
| Model | Code name | Architecture |
|-------|-----------|--------------|
| ResNet-50 | `resnet_50` | Residual network, ImageNet pre-trained |
| EfficientNet-B0 | `efficientnet_b0` | Compound-scaled CNN, ImageNet pre-trained |
| RepVGG-B0 | `repvgg_b0` | Re-parameterized VGG-style CNN |

### Audio
| Model | Code name | Input | Architecture |
|-------|-----------|-------|--------------|
| AudioNet | `audionet` | Waveform `(B, 1, T)` | 1-D CNN |
| Res1dNet31 | `res1dnet31` | Waveform `(B, T)` | 1-D ResNet |
| AlexNet (audio) | `alexnet` | Spectrogram `(B, 1, F, T)` | 2-D CNN |
| CNN14 | `cnn14` | Log-mel `(B, 1, mel, time)` | 14-layer CNN (AudioSet pre-trained) |

### EEG
| Model | Code name | Architecture |
|-------|-----------|--------------|
| EEGNet | `eegnet` | Compact depthwise separable CNN |
| InterpretableCNN | `icnn` | Temporally-interpretable CNN |
| SCCNet | `sccnet` | Spatial-component-wise CNN |

For SSVEP, frequency-adapted variants `EEGNet_SSVEP` and `InterpretableCNN_SSVEP` are used.

---

## Attribution Methods

### Image
| Method | Code |
|--------|------|
| Grad-CAM | `gradcam` |
| Grad-CAM++ | `gradcampp` |
| Score-CAM | `scorecam` |
| SmoothGrad-CAM++ | `smoothgradcampp` |
| Gradient | `gradient` |
| SmoothGrad | `smoothgrad` |
| Gradient × Input | `gradientxinput` |
| Integrated Gradients | `integrad` |
| Random | `random` |

Image saliency maps are aggregated across RGB channels using absolute values.

### Audio & EEG
Both signed and unsigned (absolute-value) variants are evaluated. Signed attributions preserve contribution direction; unsigned (abs) variants emphasize magnitude.

| Method | Signed code | Unsigned code |
|--------|-------------|---------------|
| Gradient | `gradient` / `GD` | `gradient_abs` / `GDA` |
| Gradient × Input | `gradinput` / `GI` | `gradinput_abs` / `GIA` |
| SmoothGrad | `smoothgrad` / `SG` | `smoothgrad_abs` / `SGA` |
| Integrated Gradients | `integrad` / `IG` | `integrad_abs` / `IGA` |
| SmoothGrad² | — | `smoothgrad_sq` / `SS` |
| VarGrad | — | `vargrad` / `VG` |
| Random | `random` | — |

---

## Running Experiments

### Requirements

```bash
# Image
pip install torch torchvision timm captum numpy scipy matplotlib tqdm Pillow

# Audio (additional)
pip install torchaudio librosa soundfile h5py

# EEG (additional)
pip install mne
```

### Image

```bash
cd image

# Brain Tumor MRI / ImageNet / Oxford-IIIT Pet — all via test_image.py
python experiment/test_image.py --dataset brain_mri  --model resnet_50       --expl_method gradcam         --mask_type pgd  --mode morf --n_steps 20
python experiment/test_image.py --dataset imagenet   --model efficientnet_b0 --expl_method smoothgradcampp --mask_type road
python experiment/test_image.py --dataset oxford_pet --model repvgg_b0       --expl_method gradcampp       --mask_type pgd

# Compute AOC / ABC / AUC
python experiment/area.py \
  --root image/ \
  --datasets brain_mri imagenet oxford_pet \
  --models resnet_50 efficientnet_b0 repvgg_b0 \
  --masks zero pgd road

# MoRF–LeRF Spearman consistency
python experiment/spearman.py \
  --root image/ \
  --datasets brain_mri imagenet oxford_pet \
  --models resnet_50 efficientnet_b0 repvgg_b0 \
  --masks zero pgd road
```

### Audio

```bash
cd audio

# Waveform MoRF/LeRF (rank-based)
python experiment/test_waveform.py \
  --dataset audiomnist \        # audiomnist | esc50 | msos
  --model audionet \            # audionet | res1dnet31
  --mask_type pgd \             # zero | pgd | road
  --expl_method gradient_abs \
  --n_steps 20

# Waveform MoRF/LeRF (interval-based)
python experiment/test_waveform_interval.py \
  --dataset esc50 \
  --model res1dnet31 \
  --fold 1 \                    # ESC-50 fold (1–5)
  --mask_type road \
  --expl_method smoothgrad_abs

# Spectrogram / log-mel MoRF/LeRF
python experiment/test_spectrogram.py \
  --dataset msos \
  --model cnn14 \               # alexnet | cnn14
  --mask_type pgd \
  --expl_method gradcam

# Compute AOC / ABC / AUC
python experiment/area.py \
  --root audio/ \
  --datasets audiomnist esc50 msos \
  --models audionet res1dnet31 alexnet cnn14 \
  --masks zero pgd road

# MoRF–LeRF Spearman consistency
python experiment/spearman.py \
  --root audio/ \
  --datasets audiomnist esc50 msos \
  --models audionet res1dnet31 alexnet cnn14 \
  --masks zero pgd road
```

### EEG

EEG scripts evaluate three feature axes (channel / frequency / time-segment), each with three masking variants:

| | Zero | AIM (adversarial) | mdROAD (imputation) |
|---|---|---|---|
| **Channel** | `chzero_test.py` | `chae_test.py` | `chroad_test.py` |
| **Frequency** | `fqzero_test.py` | `fqae_test.py` | `fqroad_test.py` |
| **Time-segment** | `tszero_test.py` | `tsae_test.py` | `tsroad_test.py` |

**Before running:** edit the two path constants in `eeg/experiment_utils/eeg_config.py`:

```python
# eeg/experiment_utils/eeg_config.py
IRISHSIEH_DIR = "/path/to/pretrained_models_and_explanations"
SAVE_DIR      = "/path/to/output_directory"
```

That is the only file you need to edit. Dataset selection is a CLI argument (`--dataname`).

```bash
cd eeg

# Channel masking — run once per dataset × repeat
python experiment/chzero_test.py --dataname ERN  --abs_saliency 1 --rep 0
python experiment/chae_test.py   --dataname ERN  --abs_saliency 1 --rep 0
python experiment/chroad_test.py --dataname MI   --abs_saliency 1 --rep 0

# Frequency masking
python experiment/fqzero_test.py --dataname SSVEP --abs_saliency 1 --rep 1
python experiment/fqae_test.py   --dataname ERN   --abs_saliency 1 --rep 0
python experiment/fqroad_test.py --dataname SSVEP --abs_saliency 1 --rep 1

# Time-segment masking
python experiment/tszero_test.py --dataname MI  --abs_saliency 1 --rep 0
python experiment/tsae_test.py   --dataname MI  --abs_saliency 1 --rep 0
python experiment/tsroad_test.py --dataname ERN --abs_saliency 1 --rep 0
```

To run all 5 repeats for a single script:

```bash
for rep in 0 1 2 3 4; do
  python experiment/chae_test.py --dataname ERN --abs_saliency 1 --rep $rep
done
```

> **Note:** SSVEP data is available for repeats 1 and 2 only; pass `--rep 1` or `--rep 2` for SSVEP.

**Arguments:**

| Argument | Values | Description |
|----------|--------|-------------|
| `--dataname` | `MI` \| `ERN` \| `SSVEP` | Dataset to evaluate (required) |
| `--abs_saliency` | `0` \| `1` | `1` = use absolute saliency values |
| `--rep` | `0`–`4` | Repeat / seed index (required for all scripts) |

---

## Data and Checkpoint Setup

### Image

```
image/experiment_utils/
├── data/
│   ├── Brain_MRI_Tumor/Testing/    # ImageFolder structure
│   ├── ImageNet/val/               # ILSVRC validation set
│   └── Oxford_Pet/                 # images/ + annotations/
└── checkpoints/
    ├── brain_mri/<model>/best_model.pth
    ├── imagenet/<model>/best_model.pth
    └── oxford_pet/<model>/best_model.pth
```

Pre-computed saliency maps:
```
image/expl_image_<dataset>/<model>/<method>.npy
```

### Audio

```
audio/experiment_utils/
├── data/
│   ├── audiomnist/preprocessed_data/
│   ├── ESC50/preprocessed_data/
│   └── MSoS/preprocessed_data/
└── checkpoints/
    ├── audiomnist/<model>/best_model.pth
    ├── esc50/<model>/fold_<N>/best_model.pth
    └── msos/<model>/best_model.pth
```

Pre-computed saliency maps:
```
audio/expl_<dataset>/<model>/[fold_<N>/]<method>.npy
```

### EEG

```
<IRISHSIEH_DIR>/
├── models/<DATASET>/bests_repeat<N>/
│   └── sub<S>-<model>.pth          # subject-specific checkpoints
└── repeat<N>/
    ├── ae/<DATASET>/<model>/
    │   └── sub<S>.npy              # pre-computed PGD adversarial examples
    └── expl/<DATASET>/<model>/
        └── sub<S>_<method>.npy     # pre-computed saliency maps

Raw .mat files (per-subject):
  MI/SMR:  BCIC_S<NN>_E.mat
  ERN:     Data_S<NN>_Sess.mat
  SSVEP:   U0<NN>.mat
```

---

## Evaluation Metrics

Let $\text{Acc}(x_k^M)$ and $\text{Acc}(x_k^L)$ denote model accuracy after masking the $k$ most / least important features, $x_0$ the unmasked input, and $x_{\text{allmasked}}$ the fully-masked input (all features replaced):

$$\text{AOC} = \frac{1}{K}\sum_{k=1}^{K} \frac{\text{Acc}(x_0) - \text{Acc}(x_k^M)}{\text{Acc}(x_0) - \text{Acc}(x_{\text{allmasked}})}$$

$$\text{ABC} = \frac{1}{K}\sum_{k=1}^{K} \frac{\text{Acc}(x_k^L) - \text{Acc}(x_k^M)}{\text{Acc}(x_0) - \text{Acc}(x_{\text{allmasked}})}$$

$$\text{AUC} = \frac{1}{K}\sum_{k=1}^{K} \frac{\text{Acc}(x_k^L) - \text{Acc}(x_{\text{allmasked}})}{\text{Acc}(x_0) - \text{Acc}(x_{\text{allmasked}})}$$

Higher AOC → stronger MoRF degradation (saliency identifies the most-critical features). Higher AUC → stronger LeRF preservation. Higher ABC → greater MoRF–LeRF separation. All indicate more faithful explanations.

MoRF–LeRF Spearman $\rho$ between per-method rankings measures masking-operator reliability. Random-attribution ABC near zero is a sanity check for masking bias.

---

## Compute

All experiments were conducted on NVIDIA RTX 4090 GPUs with CUDA acceleration. AIM requires one additional PGD forward/backward pass per input to generate the adversarial counterpart; all other steps are shared with zeroing and mdROAD.

---

## Acknowledgements

This codebase extends the [ROAD framework](https://github.com/tleemann/road_evaluation) (Rong et al., 2022) and builds on [Captum](https://captum.ai/) for attribution computation. The MFBB temporal imputation follows Friedrich et al. (2020). EEG models are based on [EEGNet](https://github.com/vlawhern/arl-eegmodels) (Lawhern et al., 2018), [SCCNet](https://ieeexplore.ieee.org/document/8717134) (Wei et al., 2019), and [InterpretableCNN](https://ieeexplore.ieee.org/document/9737231) (Cui et al., 2022).
