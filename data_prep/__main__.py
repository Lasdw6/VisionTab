"""Allow running data_prep as a module: python -m data_prep --config configs/training_config.yaml"""
from .prepare_dataset import main

main()
