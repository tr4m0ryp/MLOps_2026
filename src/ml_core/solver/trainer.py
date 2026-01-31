import time
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils import ExperimentTracker, setup_logger


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
        device: str,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.logger = setup_logger("Trainer")

        # Define Loss Function (CrossEntropy for classification)
        self.criterion = nn.CrossEntropyLoss()

        # Initialize ExperimentTracker
        self.tracker = ExperimentTracker(
            experiment_name=config.get("experiment_name", "experiment"),
            config=config,
            base_dir=config.get("training", {}).get("save_dir", "experiments/results"),
        )

        # Best validation loss for checkpointing
        self.best_val_loss = float("inf")

    def train_epoch(
        self, dataloader: DataLoader, epoch_idx: int
    ) -> Tuple[float, float]:
        """
        Train the model for one epoch.

        Args:
            dataloader: Training data loader
            epoch_idx: Current epoch index

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(
            dataloader, desc=f"Epoch {epoch_idx + 1} [Train]", leave=False
        )

        for batch_idx, (images, labels) in enumerate(progress_bar):
            # Move data to device
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            progress_bar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{100 * correct / total:.2f}%"}
            )

        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self, dataloader: DataLoader, epoch_idx: int) -> Tuple[float, float]:
        """
        Validate the model.

        Args:
            dataloader: Validation data loader
            epoch_idx: Current epoch index

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(
            dataloader, desc=f"Epoch {epoch_idx + 1} [Val]", leave=False
        )

        for images, labels in progress_bar:
            # Move data to device
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            progress_bar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{100 * correct / total:.2f}%"}
            )

        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            val_loss: Validation loss
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
        }

        # Save latest checkpoint
        checkpoint_path = self.tracker.get_checkpoint_path("checkpoint_latest.pt")
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.tracker.get_checkpoint_path("checkpoint_best.pt")
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model with val_loss: {val_loss:.4f}")

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        epochs = self.config["training"]["epochs"]
        learning_rate = self.config["training"]["learning_rate"]

        self.logger.info(f"Starting training for {epochs} epochs...")
        self.logger.info(f"Results will be saved to: {self.tracker.run_dir}")

        try:
            for epoch in range(epochs):
                epoch_start = time.time()

                # Train
                train_loss, train_acc = self.train_epoch(train_loader, epoch)

                # Validate
                val_loss, val_acc = self.validate(val_loader, epoch)

                epoch_time = time.time() - epoch_start

                # Check if this is the best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss

                # Log metrics
                metrics = {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "learning_rate": learning_rate,
                    "epoch_time": epoch_time,
                }
                self.tracker.log_metrics(epoch + 1, metrics)

                # Save checkpoint
                self.save_checkpoint(epoch + 1, val_loss, is_best)

                # Print epoch summary
                self.logger.info(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
                    f"Time: {epoch_time:.2f}s"
                )

        finally:
            # Ensure tracker is closed properly
            self.tracker.close()
            self.logger.info("Training completed!")
            self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
