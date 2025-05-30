"""Experiment state management."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ExperimentState:
    """Manage experiment state and checkpointing.
    
    This class provides utilities for saving and loading experiment
    state, enabling resume functionality and result persistence.
    """
    
    def __init__(self, experiment_id: str, state_dir: Optional[str] = None):
        """Initialize state manager.
        
        Args:
            experiment_id: Unique experiment identifier
            state_dir: Directory to store state files (default: ./states)
        """
        self.experiment_id = experiment_id
        self.state_dir = Path(state_dir or "./states")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        self.state = {
            'experiment_id': experiment_id,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'checkpoints': {},
            'metadata': {},
            'results': {}
        }
        
        # Try to load existing state
        self._load_state()
        
    def _get_state_path(self) -> Path:
        """Get path to state file."""
        return self.state_dir / f"{self.experiment_id}_state.json"
        
    def _load_state(self) -> None:
        """Load existing state if available."""
        state_path = self._get_state_path()
        
        if state_path.exists():
            try:
                with open(state_path, 'r') as f:
                    self.state = json.load(f)
                logger.info(f"Loaded existing state for {self.experiment_id}")
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
                
    def save(self) -> None:
        """Save current state to file."""
        self.state['updated_at'] = datetime.now().isoformat()
        
        state_path = self._get_state_path()
        with open(state_path, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)
            
        logger.debug(f"Saved state to {state_path}")
        
    def checkpoint(self, name: str, data: Dict[str, Any]) -> None:
        """Create a checkpoint.
        
        Args:
            name: Checkpoint name
            data: Data to checkpoint
        """
        checkpoint_file = self.state_dir / f"{self.experiment_id}_{name}.pkl"
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)
            
        self.state['checkpoints'][name] = {
            'path': str(checkpoint_file),
            'created_at': datetime.now().isoformat(),
            'keys': list(data.keys()) if isinstance(data, dict) else None
        }
        
        self.save()
        logger.info(f"Created checkpoint: {name}")
        
    def load_checkpoint(self, name: str) -> Any:
        """Load a checkpoint.
        
        Args:
            name: Checkpoint name
            
        Returns:
            Checkpointed data
            
        Raises:
            ValueError: If checkpoint doesn't exist
        """
        if name not in self.state['checkpoints']:
            raise ValueError(f"Checkpoint '{name}' not found")
            
        checkpoint_info = self.state['checkpoints'][name]
        checkpoint_file = Path(checkpoint_info['path'])
        
        if not checkpoint_file.exists():
            raise ValueError(f"Checkpoint file not found: {checkpoint_file}")
            
        with open(checkpoint_file, 'rb') as f:
            data = pickle.load(f)
            
        logger.info(f"Loaded checkpoint: {name}")
        return data
        
    def has_checkpoint(self, name: str) -> bool:
        """Check if checkpoint exists.
        
        Args:
            name: Checkpoint name
            
        Returns:
            True if checkpoint exists
        """
        if name not in self.state['checkpoints']:
            return False
            
        checkpoint_file = Path(self.state['checkpoints'][name]['path'])
        return checkpoint_file.exists()
        
    def list_checkpoints(self) -> Dict[str, Dict[str, Any]]:
        """List all checkpoints.
        
        Returns:
            Dictionary of checkpoint information
        """
        return self.state['checkpoints'].copy()
        
    def update_metadata(self, **kwargs) -> None:
        """Update experiment metadata.
        
        Args:
            **kwargs: Metadata key-value pairs
        """
        self.state['metadata'].update(kwargs)
        self.save()
        
    def update_results(self, **kwargs) -> None:
        """Update experiment results.
        
        Args:
            **kwargs: Results key-value pairs
        """
        self.state['results'].update(kwargs)
        self.save()
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get experiment metadata."""
        return self.state['metadata'].copy()
        
    def get_results(self) -> Dict[str, Any]:
        """Get experiment results."""
        return self.state['results'].copy()
        
    def cleanup(self, keep_final: bool = True) -> None:
        """Clean up state files.
        
        Args:
            keep_final: Whether to keep the final state file
        """
        # Remove checkpoint files
        for checkpoint_info in self.state['checkpoints'].values():
            checkpoint_file = Path(checkpoint_info['path'])
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                logger.debug(f"Removed checkpoint: {checkpoint_file}")
                
        # Remove state file if requested
        if not keep_final:
            state_path = self._get_state_path()
            if state_path.exists():
                state_path.unlink()
                logger.debug(f"Removed state file: {state_path}")
                
        logger.info(f"Cleaned up state for {self.experiment_id}")