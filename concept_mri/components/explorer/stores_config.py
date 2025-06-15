"""
Store configurations and data schemas for Network Explorer.

This module defines the structure of all data stores used for
cross-component communication in the Network Explorer.
"""

from typing import TypedDict, Dict, List, Any, Optional


class ClusterData(TypedDict):
    """Structure for cluster information."""
    n_clusters: int
    labels: List[int]
    centroids: Optional[List[List[float]]]
    sizes: Dict[int, int]
    metrics: Dict[str, float]


class PathData(TypedDict):
    """Structure for path information."""
    path_id: str
    sequence: List[str]  # List of cluster IDs like ["L0_C1", "L1_C2", ...]
    frequency: int
    samples: List[int]  # Sample indices following this path
    stability: float
    pattern: str  # Pattern type (e.g., "stable", "transitional", "fragmented")
    transitions: List[Dict[str, Any]]


class WindowConfig(TypedDict):
    """Structure for window configuration."""
    windows: Dict[str, Dict[str, int]]  # e.g., {"early": {"start": 0, "end": 3}}
    current_window: str  # Currently selected window
    auto_detect: bool  # Whether to auto-detect windows


class SelectionData(TypedDict):
    """Structure for current selection."""
    entity_type: Optional[str]  # "cluster", "path", "sample", or None
    entity_id: Optional[str]
    entity_data: Optional[Dict[str, Any]]
    source_component: Optional[str]  # Which component triggered the selection


class HighlightData(TypedDict):
    """Structure for cross-component highlighting."""
    highlight_type: Optional[str]  # "path", "cluster", "sample"
    highlight_ids: List[str]  # IDs to highlight
    source_component: Optional[str]
    color: Optional[str]  # Highlight color


class ClusteringResultsStore(TypedDict):
    """Main clustering results store structure."""
    completed: bool
    model_name: str
    timestamp: str
    
    # Clustering data per layer
    clusters_per_layer: Dict[str, ClusterData]
    
    # Path analysis results
    paths: Dict[str, PathData]  # path_id -> PathData
    total_samples: int
    unique_paths: int
    
    # Cluster labels (from LLM)
    cluster_labels: Dict[str, str]  # cluster_id -> semantic_label
    
    # Metrics
    metrics: Dict[str, Any]
    
    # Optional hierarchy results
    hierarchy_results: Optional[Dict[str, Any]]


class PathsAnalysisStore(TypedDict):
    """Analyzed paths with patterns and insights."""
    paths: List[PathData]
    patterns: Dict[str, List[str]]  # pattern_type -> list of path_ids
    statistics: Dict[str, Any]
    llm_analysis: Optional[Dict[str, Any]]


# Store IDs used in the application
STORE_IDS = {
    "clustering_results": "network-explorer-clustering-results-store",
    "selection": "network-explorer-selection-store",
    "window_config": "network-explorer-window-config-store",
    "paths_analysis": "network-explorer-paths-analysis-store",
    "highlight": "network-explorer-highlight-store",
    "llm_analysis_trigger": "llm-analysis-trigger-store",
    "comparison": "network-explorer-comparison-store"
}


# Default store values
DEFAULT_STORES = {
    "clustering_results": {
        "completed": False,
        "model_name": "",
        "timestamp": "",
        "clusters_per_layer": {},
        "paths": {},
        "total_samples": 0,
        "unique_paths": 0,
        "cluster_labels": {},
        "metrics": {},
        "hierarchy_results": None
    },
    "selection": {
        "entity_type": None,
        "entity_id": None,
        "entity_data": None,
        "source_component": None
    },
    "window_config": {
        "windows": {
            "early": {"start": 0, "end": 3},
            "middle": {"start": 4, "end": 7},
            "late": {"start": 8, "end": 11}
        },
        "current_window": "full",
        "auto_detect": True
    },
    "paths_analysis": {
        "paths": [],
        "patterns": {},
        "statistics": {},
        "llm_analysis": None
    },
    "highlight": {
        "highlight_type": None,
        "highlight_ids": [],
        "source_component": None,
        "color": None
    },
    "llm_analysis_trigger": None,
    "comparison": {
        "active": False,
        "entities": []
    }
}


def create_stores():
    """Create all dcc.Store components for Network Explorer.
    
    Returns:
        List of dcc.Store components
    """
    from dash import dcc
    
    stores = []
    for store_name, store_id in STORE_IDS.items():
        stores.append(
            dcc.Store(
                id=store_id,
                data=DEFAULT_STORES.get(store_name, None),
                storage_type='memory'  # Use session storage for persistence across page refreshes
            )
        )
    
    return stores


def get_store_id(store_name: str) -> str:
    """Get the store ID for a given store name.
    
    Args:
        store_name: Name of the store (e.g., "clustering_results")
        
    Returns:
        Store ID string
        
    Raises:
        ValueError: If store name is not recognized
    """
    if store_name not in STORE_IDS:
        raise ValueError(f"Unknown store name: {store_name}. Available stores: {list(STORE_IDS.keys())}")
    return STORE_IDS[store_name]