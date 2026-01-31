import csv
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
import yaml

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class ExperimentTracker:
    def __init__(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        base_dir: str = "experiments/results",
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Save config to yaml in run_dir
        config_path = self.run_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Initialize CSV for metrics logging
        self.csv_path = self.run_dir / "metrics.csv"
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        # Header with all metrics we want to track
        self.csv_writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "learning_rate",
            "epoch_time"
        ])

        # Initialize TensorBoard if available
        self.tb_writer: Optional[SummaryWriter] = None
        if TENSORBOARD_AVAILABLE:
            self.tb_writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))

    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """
        Writes metrics to CSV and TensorBoard.

        Args:
            epoch: Current epoch number
            metrics: Dictionary containing metric names and values
        """
        # Write to CSV
        row = [
            epoch,
            metrics.get("train_loss", 0.0),
            metrics.get("train_accuracy", 0.0),
            metrics.get("val_loss", 0.0),
            metrics.get("val_accuracy", 0.0),
            metrics.get("learning_rate", 0.0),
            metrics.get("epoch_time", 0.0),
        ]
        self.csv_writer.writerow(row)
        self.csv_file.flush()

        # Log to TensorBoard
        if self.tb_writer is not None:
            for name, value in metrics.items():
                self.tb_writer.add_scalar(name, value, epoch)

    def log_model_graph(self, model, input_sample):
        """Log model architecture to TensorBoard."""
        if self.tb_writer is not None:
            try:
                self.tb_writer.add_graph(model, input_sample)
            except Exception:
                pass  # Graph logging can fail for some model types

    def get_checkpoint_path(self, filename: str) -> str:
        return str(self.run_dir / filename)

    def close(self):
        self.csv_file.close()
        if self.tb_writer is not None:
            self.tb_writer.close()
