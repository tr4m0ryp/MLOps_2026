import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training metrics.")
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def load_data(file_path: Path) -> pd.DataFrame:
    """Load CSV file into a Pandas DataFrame."""
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    return pd.read_csv(file_path)


def setup_style():
    """Set seaborn and matplotlib style for publication-quality plots."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 11,
    })


def plot_metrics(df: pd.DataFrame, output_path: Optional[Path]):
    """
    Generate and save plots for Loss, Accuracy, and Learning Rate.

    Args:
        df: DataFrame containing training metrics
        output_path: Directory to save plots (if None, displays interactively)
    """
    if df is None or df.empty:
        print("No data to plot.")
        return

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Train/Val Loss
    ax1 = axes[0, 0]
    if "train_loss" in df.columns and "val_loss" in df.columns:
        ax1.plot(df["epoch"], df["train_loss"], "b-o", label="Train Loss", markersize=4)
        ax1.plot(df["epoch"], df["val_loss"], "r-s", label="Val Loss", markersize=4)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "Loss data not available", ha="center", va="center")
        ax1.set_title("Loss")

    # Plot 2: Train/Val Accuracy
    ax2 = axes[0, 1]
    if "train_accuracy" in df.columns and "val_accuracy" in df.columns:
        ax2.plot(
            df["epoch"], df["train_accuracy"], "b-o", label="Train Accuracy", markersize=4
        )
        ax2.plot(
            df["epoch"], df["val_accuracy"], "r-s", label="Val Accuracy", markersize=4
        )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Training and Validation Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Accuracy data not available", ha="center", va="center")
        ax2.set_title("Accuracy")

    # Plot 3: Learning Rate (if available)
    ax3 = axes[1, 0]
    if "learning_rate" in df.columns:
        ax3.plot(df["epoch"], df["learning_rate"], "g-^", markersize=4)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Learning Rate")
        ax3.set_title("Learning Rate Schedule")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "Learning rate data not available", ha="center", va="center")
        ax3.set_title("Learning Rate")

    # Plot 4: Epoch Time (if available)
    ax4 = axes[1, 1]
    if "epoch_time" in df.columns:
        ax4.bar(df["epoch"], df["epoch_time"], color="purple", alpha=0.7)
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Time (seconds)")
        ax4.set_title("Training Time per Epoch")
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "Timing data not available", ha="center", va="center")
        ax4.set_title("Epoch Time")

    plt.tight_layout()

    # Save or show
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / "training_metrics.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    args = parse_args()
    setup_style()
    df = load_data(args.input_csv)
    print(f"Loaded {len(df)} rows from {args.input_csv}")
    print(f"Columns: {list(df.columns)}")
    plot_metrics(df, args.output_dir)


if __name__ == "__main__":
    main()
