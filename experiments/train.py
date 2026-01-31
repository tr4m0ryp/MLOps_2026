import argparse

import torch
import torch.optim as optim

from ml_core.data import get_dataloaders
from ml_core.models import MLP
from ml_core.solver import Trainer
from ml_core.utils import load_config, seed_everything, setup_logger

logger = setup_logger("Experiment_Runner")


def main(args):
    # 1. Load Config & Set Seed
    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)

    seed = config.get("seed", 42)
    seed_everything(seed)
    logger.info(f"Random seed set to: {seed}")

    # 2. Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 3. Data
    logger.info("Loading data...")
    train_loader, val_loader = get_dataloaders(config["data"])
    logger.info(
        f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
    )

    # 4. Model
    input_shape = config["data"]["input_shape"]
    input_dim = input_shape[0] * input_shape[1] * input_shape[2]  # C * H * W
    hidden_units = config["model"]["hidden_units"]
    dropout_rate = config["model"]["dropout_rate"]
    num_classes = config["model"]["num_classes"]

    model = MLP(
        input_dim=input_dim,
        hidden_units=hidden_units,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )
    logger.info(f"Model created: MLP with {sum(p.numel() for p in model.parameters())} parameters")

    # 5. Optimizer
    learning_rate = config["training"]["learning_rate"]
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    logger.info(f"Optimizer: Adam with lr={learning_rate}")

    # 6. Trainer & Fit
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config,
        device=device,
    )
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Simple MLP on PCAM")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config yaml",
    )
    args = parser.parse_args()

    main(args)
