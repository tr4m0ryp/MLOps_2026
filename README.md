# MLOps 2026 - Medical Image Classification

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)

A production-ready MLOps template for training neural networks on medical imaging data (PatchCamelyon/PCAM dataset). This repository demonstrates best practices in machine learning engineering including modular architecture, reproducibility, automated testing, and experiment tracking.

---

## Features

- **Modular Architecture**: Clean separation between data loading, models, training, and utilities
- **Reproducibility**: Seeded random states, configuration files, and experiment tracking
- **Experiment Tracking**: CSV logging and TensorBoard integration for metrics visualization
- **Automated Testing**: Comprehensive unit tests with pytest
- **Code Quality**: Pre-commit hooks with Ruff for linting and formatting
- **CI/CD**: GitHub Actions workflow for continuous integration

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/SURF-ML/MLOps_2026.git
cd MLOps_2026
```

### 2. Create Virtual Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
# Install package in editable mode with all dependencies
pip install -e .

# Install pre-commit hooks
pre-commit install
```

### 4. Download the Dataset

Download the PatchCamelyon dataset from [Kaggle](https://www.kaggle.com/c/histopathologic-cancer-detection) or the official source. Place the H5 files in the `data/` directory:

```
data/
└── camelyonpatch_level_2/
    ├── camelyonpatch_level_2_split_train_x.h5
    ├── camelyonpatch_level_2_split_train_y.h5
    ├── camelyonpatch_level_2_split_valid_x.h5
    └── camelyonpatch_level_2_split_valid_y.h5
```

Update the `data_path` in `experiments/configs/train_config.yaml` to point to your data location.

### 5. Verify Installation

```bash
# Run unit tests
pytest tests/ -v
```

### 6. Train a Model

```bash
python experiments/train.py --config experiments/configs/train_config.yaml
```

---

## Project Structure

```
MLOps_2026/
├── src/ml_core/              # Core library code
│   ├── data/                 # Data loading and preprocessing
│   │   ├── loader.py         # DataLoader factory with weighted sampling
│   │   └── pcam.py           # PatchCamelyon Dataset class
│   ├── models/               # Neural network architectures
│   │   └── mlp.py            # Multi-layer Perceptron classifier
│   ├── solver/               # Training logic
│   │   └── trainer.py        # Trainer class with full training loop
│   └── utils/                # Utility functions
│       ├── logging.py        # Logger setup, config loading, seeding
│       └── tracker.py        # Experiment tracking (CSV + TensorBoard)
│
├── experiments/              # Experiment execution
│   ├── configs/              # YAML configuration files
│   │   └── train_config.yaml # Default training configuration
│   ├── results/              # Output directory (auto-generated)
│   └── train.py              # Main training entry point
│
├── scripts/                  # Utility scripts
│   ├── torch_example.py      # Standalone PyTorch example
│   ├── example_training_loop/
│   │   └── training_loop.py  # Reference training implementation
│   └── plotting/
│       └── plot_results_csv.py  # Visualization for training metrics
│
├── tests/                    # Unit tests
│   ├── test_imports.py       # Module import tests
│   ├── test_model_shapes.py  # Model architecture tests
│   └── test_data_loader.py   # Data loading tests
│
├── .github/workflows/        # CI/CD configuration
│   └── ci.yml                # GitHub Actions workflow
│
├── pyproject.toml            # Project metadata and tool configuration
├── requirements.txt          # Python dependencies
├── .pre-commit-config.yaml   # Pre-commit hook configuration
├── LICENSE                   # MIT License
└── README.md                 # This file
```

---

## Configuration

Training parameters are managed via YAML configuration files. The default configuration is located at `experiments/configs/train_config.yaml`:

```yaml
experiment_name: "pcam_mlp_baseline"
seed: 42

data:
  dataset_type: "pcam"
  data_path: "./data/camelyonpatch_level_2"
  input_shape: [3, 96, 96]
  batch_size: 32
  num_workers: 2

model:
  hidden_units: [64, 32]
  dropout_rate: 0.2
  num_classes: 2

training:
  epochs: 5
  learning_rate: 0.001
  save_dir: "./experiments/results"
```

---

## Usage Examples

### Training with Custom Configuration

```bash
# Copy and modify the default config
cp experiments/configs/train_config.yaml experiments/configs/my_experiment.yaml
# Edit my_experiment.yaml with your parameters
python experiments/train.py --config experiments/configs/my_experiment.yaml
```

### Visualizing Training Results

After training, visualize metrics from the generated CSV:

```bash
python scripts/plotting/plot_results_csv.py \
    --input_csv experiments/results/<experiment_name>/metrics.csv \
    --output_dir experiments/results/<experiment_name>/plots
```

### TensorBoard Monitoring

If TensorBoard is installed, view real-time metrics:

```bash
tensorboard --logdir experiments/results/
```

---

## Testing

Run the full test suite:

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_model_shapes.py -v

# With coverage report
pytest tests/ --cov=ml_core --cov-report=html
```

---

## Code Quality

This project uses Ruff for linting and formatting:

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check . --fix

# Format code
ruff format .
```

Pre-commit hooks run automatically on `git commit`. To run manually:

```bash
pre-commit run --all-files
```

---

## Model Architecture

The default model is a 3-layer Multi-Layer Perceptron (MLP):

| Layer | Description |
|-------|-------------|
| Input | Flatten 3x96x96 = 27,648 features |
| Hidden 1 | Fully connected (64 units) + ReLU + Dropout |
| Hidden 2 | Fully connected (32 units) + ReLU + Dropout |
| Output | Fully connected (2 classes) |

---

## Experiment Outputs

Each training run creates a timestamped directory in `experiments/results/` containing:

```
experiments/results/<experiment_name>_<timestamp>/
├── config.yaml           # Copy of training configuration
├── metrics.csv           # Epoch-wise metrics (loss, accuracy, time)
├── checkpoint_latest.pt  # Most recent model checkpoint
├── checkpoint_best.pt    # Best model (lowest validation loss)
└── tensorboard/          # TensorBoard log files
```

---

## Dependencies

Core dependencies (see `requirements.txt` for full list):

- Python >= 3.10
- PyTorch >= 2.0
- NumPy
- h5py (for reading PCAM dataset)
- PyYAML (configuration management)
- tqdm (progress bars)
- matplotlib, seaborn, pandas (visualization)
- pytest (testing)
- ruff (linting)

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [SURF-ML](https://github.com/SURF-ML) - Original template creators
- [PatchCamelyon Dataset](https://github.com/basveeling/pcam) - Medical imaging dataset
- UvA Bachelor AI Course - MLOps curriculum
