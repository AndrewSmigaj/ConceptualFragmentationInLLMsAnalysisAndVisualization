import os, datetime
from typing import List, Optional, Union
from pathlib import Path
from concept_fragmentation.config import RESULTS_DIR

def build_experiment_dir(dataset: str,
                         config_id: str = "baseline",
                         seed: int = 0,
                         timestamp: Optional[str] = None) -> str:
    """
    Canonical path:
      RESULTS_DIR / <config_id>s / <dataset> /
        f"{dataset}_{config_id}_seed{seed}_{timestamp}"
    Example:
      D:/concept_fragmentation_results/baselines/heart/
           heart_baseline_seed0_20250519_154501
    """
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    root = os.path.join(RESULTS_DIR, f"{config_id}s", dataset)  # baselines/, regularized/
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, f"{dataset}_{config_id}_seed{seed}_{timestamp}")


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path: Path to the directory
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path