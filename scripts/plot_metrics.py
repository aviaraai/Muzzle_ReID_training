import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
METRICS_CSV = RESULTS_DIR / "metrics.csv"
OUT_IMG = RESULTS_DIR / "loss_curves.png"

def main():
    if not METRICS_CSV.exists():
        print(f"Metrics file not found: {METRICS_CSV}")
        return

    df = pd.read_csv(METRICS_CSV)

    if df.empty:
        print("Metrics file is empty.")
        return

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Train & Val Loss
    axs[0].plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o", markersize=4)
    axs[0].plot(df["epoch"], df["val_loss"], label="Val Loss", marker="o", markersize=4)
    axs[0].set_title("Training & Validation Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True, linestyle="--", alpha=0.6)

    # Plot 2: Top-1 & Top-5 Accuracy
    axs[1].plot(df["epoch"], df["top1"] * 100, label="Top-1 Acc", marker="s", markersize=4)
    axs[1].plot(df["epoch"], df["top5"] * 100, label="Top-5 Acc", marker="s", markersize=4)
    axs[1].set_title("Retrieval Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].legend()
    axs[1].grid(True, linestyle="--", alpha=0.6)

    # Plot 3: Genuine & Impostor Means, and Gap
    axs[2].plot(df["epoch"], df["genuine_mean"], label="Genuine Mean", marker="^", markersize=4)
    axs[2].plot(df["epoch"], df["impostor_mean"], label="Impostor Mean", marker="v", markersize=4)
    axs[2].plot(df["epoch"], df["gap"], label="Gap", marker="D", markersize=4, linestyle="--", color="purple")
    axs[2].set_title("Genuine/Impostor Similarities")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Cosine Similarity")
    axs[2].legend()
    axs[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(OUT_IMG, dpi=200)
    print(f"Saved loss curves to {OUT_IMG}")

if __name__ == "__main__":
    main()
