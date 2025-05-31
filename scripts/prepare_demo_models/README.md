# Demo Model Preparation Scripts

This directory contains scripts for training feedforward neural networks that can be analyzed by Concept MRI.

## Purpose

These scripts are **separate** from Concept MRI itself. They are used to:
1. Train optimal models using hyperparameter optimization
2. Create intentionally suboptimal models to demonstrate various issues
3. Save models in a format compatible with Concept MRI

## Structure

- `train_demo_models.py` - Main training script with CLI
- `model_trainer.py` - Base training class
- `model_architectures.py` - Neural network definitions
- `model_presets.py` - Predefined configurations
- `hyperparameter_search.py` - Optuna integration
- `configs/` - Configuration files
- `utils/` - Helper utilities

## Usage

```bash
# Train optimal model
python train_demo_models.py --dataset titanic --variant optimal

# Train with specific architecture
python train_demo_models.py --dataset heart_disease --variant bottleneck

# Custom configuration
python train_demo_models.py --dataset titanic --variant custom --config configs/my_config.json
```