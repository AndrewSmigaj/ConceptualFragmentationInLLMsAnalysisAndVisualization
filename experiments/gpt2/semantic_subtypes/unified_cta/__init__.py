"""
Unified CTA (Concept Trajectory Analysis) Implementation

A two-tier clustering approach for analyzing GPT-2's semantic organization:
- Macro clusters: Structural organization for trajectory analysis
- Micro clusters: Explainable sub-clusters for interpretability
"""

__version__ = "0.1.0"

# Import main components for easier access
from . import preprocessing
from . import clustering
from . import explainability
from . import paths
from . import llm
from . import diagnostics
from . import visualization

__all__ = [
    "preprocessing",
    "clustering", 
    "explainability",
    "paths",
    "llm",
    "diagnostics",
    "visualization"
]