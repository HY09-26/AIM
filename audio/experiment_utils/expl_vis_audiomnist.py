import numpy as np
import h5py
import argparse
import os
import matplotlib.pyplot as plt
import librosa.display

############################################
# Utility
############################################
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


############################################
# Visualization for AlexNet (Spectrogram)
############################################
def visualize_spectrogram_saliency(input_spec, saliency, out_prefix):
    """
    input_spec: (1, 227, 227)
    saliency:   (1, 227, 227)
    """

    input_spec = input_spec.squeeze()
    saliency = saliency.squeeze()

    # Normalize saliency
    saliency_norm = saliency / (np.max(np.abs(saliency)) + 1e-6)

    # ----- Heatmap -----
    plt.figure(figsize=(6, 5))
    plt.imshow(saliency_norm, cmap="jet", aspect="auto", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("Saliency Heatmap")
    plt.tight_layout()
    heat_path = f"{out_prefix}_heatmap.png"
    plt.savefig(heat_path, dpi=200)
    plt.close()

    # ----- Overlay -----
    plt.figure(figsize=(6, 5))
    plt.imshow(input_spec, cmap="gray", aspect="auto")
    plt.imshow(saliency_norm, cmap="jet", alpha=0.5, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("Input + Saliency Overlay")
    plt.tight_layout()
    overlay_path = f"{out_prefix}_overlay.png"
    plt.savefig(overlay_path, dpi=200)
    plt.close()

    print(f"Saved AlexNet heatmap → {heat_path}")
    print(f"Saved AlexNet overlay → {overlay_path}")


############################################
# Visualization for AudioNet (Waveform)
############################################
def visualize_waveform_saliency(waveform, saliency, out_prefix):
    """
    waveform: (8000,)
    saliency: (8000,)
    """

    waveform = waveform.flatten()
    saliency = saliency.flatten()

    # Normalize saliency
    saliency_norm = saliency / (np.max(np.abs(saliency)) + 1e-6)
    #saliency_norm = saliency

    t = np.arange(len(waveform)) / 8000.0
    #t = np.arange(len(waveform))

    # ----- Heatmap-like Line Plot -----
    plt.figure(figsize=(10, 4))
    plt.plot(t, saliency_norm, linewidth=1)
    plt.title("1D Saliency (AudioNet)")
    plt.tight_layout()
    heat_path = f"{out_prefix}_heatmap.png"
    plt.savefig(heat_path, dpi=200)
    plt.close()

    # ----- Overlay (Waveform + Saliency) -----
    plt.figure(figsize=(10, 4))
    plt.plot(t, waveform, color="black", alpha=0.6, label="waveform")
    plt.plot(t, saliency_norm, color="red", alpha=0.7, label="saliency")
    plt.legend()
    plt.title("Waveform + Saliency Overlay (AudioNet)")
    plt.tight_layout()
    overlay_path = f"{out_prefix}_overlay.png"
    plt.savefig(overlay_path, dpi=200)
    plt.close()

    print(f"Saved AudioNet heatmap → {heat_path}")
    print(f"Saved AudioNet overlay → {overlay_path}")


############################################
# Main
############################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["alexnet", "audionet"])
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--idx", type=int, default=0,
                        help="Which test sample to visualize")
    args = parser.parse_args()

    MODEL = args.model
    METHOD = args.method
    IDX = args.idx

    ############################################
    # Load explanation file
    ############################################
    expl_path = f"./expl_audio/{MODEL}/{METHOD}.npy"
    print(f"Loading explanation → {expl_path}")

    if not os.path.exists(expl_path):
        raise FileNotFoundError(f"Explanation file not found: {expl_path}")

    expl = np.load(expl_path)   # shape: (N, 1, H, W) or (N, 8000)
    print(f"Explanation Loaded, shape = {expl.shape}")

    ############################################
    # Load corresponding test split
    ############################################
    if MODEL == "alexnet":
        split_file = "./preprocessed_data/AlexNet_digit_0_test.txt"
    else:
        split_file = "./preprocessed_data/AudioNet_digit_0_test.txt"

    print(f"Loading test list → {split_file}")

    with open(split_file, "r") as f:
        test_list = [line.strip() for line in f.readlines()]

    if IDX >= len(test_list):
        raise IndexError(
            f"idx={IDX} out of range! test set has {len(test_list)} entries."
        )

    raw_input_path = test_list[IDX]
    print(f"Using raw input from: {raw_input_path}")

    ############################################
    # Load raw input data
    ############################################
    with h5py.File(raw_input_path, "r") as f:
        raw = np.array(f["data"])[0]   # AlexNet: (1,227,227), AudioNet: (1,1,8000 or 8000)

    ############################################
    # Output folder
    ############################################
    out_dir = "viz_output"
    ensure_dir(out_dir)
    out_prefix = os.path.join(out_dir, f"{MODEL}_{METHOD}_{IDX}")

    ############################################
    # Run Visualization
    ############################################
    if MODEL == "alexnet":
        visualize_spectrogram_saliency(raw, expl[IDX], out_prefix)

    elif MODEL == "audionet":
        # raw shape: (1,1,1,8000) → flatten to (8000,)
        waveform = raw.flatten()
        visualize_waveform_saliency(waveform, expl[IDX], out_prefix)

    print("Visualization complete!")
