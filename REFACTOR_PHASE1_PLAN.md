# Phase 1: Create Core Directory Structure and Base Interfaces

## Objective
Establish the foundational directory structure and define base interfaces that all implementations will follow.

## Design Review

### Directory Structure
```
concept_fragmentation/
├── __init__.py
├── clustering/
│   ├── __init__.py
│   ├── base.py         # Abstract base class for clusterers
│   ├── kmeans.py       # K-means implementation
│   └── paths.py        # Path extraction and analysis
├── labeling/
│   ├── __init__.py
│   ├── base.py         # Abstract base class for labelers
│   ├── consistent.py   # ConsistentLabeler implementation
│   └── semantic_purity.py  # Purity calculations
├── visualization/
│   ├── __init__.py
│   ├── base.py         # Abstract base class for visualizers
│   ├── sankey.py       # SankeyGenerator
│   ├── trajectory.py   # TrajectoryVisualizer
│   └── utils.py        # Shared utilities
├── experiments/
│   ├── __init__.py
│   ├── base.py         # BaseExperiment class
│   └── config.py       # Configuration management
└── persistence/
    ├── __init__.py
    └── state.py        # Experiment state management
```

### Base Interfaces Design

#### 1. BaseClusterer
```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np

class BaseClusterer(ABC):
    """Abstract base class for all clustering algorithms."""
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseClusterer':
        """Fit the clusterer to data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels."""
        pass
    
    @abstractmethod
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict in one step."""
        pass
    
    @property
    @abstractmethod
    def n_clusters(self) -> int:
        """Return number of clusters."""
        pass
```

#### 2. BaseLabeler
```python
class BaseLabeler(ABC):
    """Abstract base class for cluster labeling strategies."""
    
    @abstractmethod
    def label_clusters(self, 
                      cluster_data: Dict[str, Any],
                      tokens: List[str]) -> Dict[str, Dict[str, Any]]:
        """Generate labels for clusters."""
        pass
    
    @abstractmethod
    def get_label_consistency(self, labels: Dict) -> float:
        """Calculate label consistency score."""
        pass
```

#### 3. BaseVisualizer
```python
class BaseVisualizer(ABC):
    """Abstract base class for visualization components."""
    
    @abstractmethod
    def create_figure(self, data: Dict[str, Any]) -> Any:
        """Create visualization figure."""
        pass
    
    @abstractmethod
    def save_figure(self, fig: Any, output_path: str) -> None:
        """Save figure to file."""
        pass
```

### Configuration Management
```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    name: str
    model: str
    k_values: List[int]
    layers: List[int]
    output_dir: str
    random_seed: Optional[int] = 42
    
    @classmethod
    def from_yaml(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        pass
```

## Implementation Steps

1. **Create directory structure**
   - Use pathlib for path management
   - Create all __init__.py files with proper imports

2. **Implement base classes**
   - Start with abstract interfaces
   - Add type hints throughout
   - Include comprehensive docstrings

3. **Add utility functions**
   - Path validation
   - Configuration loading
   - Logging setup

4. **Create tests**
   - Test directory structure exists
   - Test base classes can be inherited
   - Test configuration loading

## Quality Checks

1. **Code Quality**
   - All code follows PEP 8
   - Type hints on all public methods
   - Docstrings follow Google style

2. **Architecture Quality**
   - Clean separation of concerns
   - No circular dependencies
   - Extensible design

3. **Testing**
   - Unit tests for each module
   - Integration test for imports
   - No hard-coded paths

## Risks and Mitigations

1. **Risk**: Breaking existing code
   - **Mitigation**: Create new structure alongside old code initially

2. **Risk**: Import path conflicts
   - **Mitigation**: Use absolute imports, update sys.path carefully

3. **Risk**: Missing functionality
   - **Mitigation**: Review all existing implementations before defining interfaces

## Success Criteria

- [x] Directory structure created
- [x] All base classes implemented
- [x] Configuration system working
- [x] Tests passing
- [x] Can import from concept_fragmentation package