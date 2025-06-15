"""
SelectionManager for coordinating state across Network Explorer components.

This module handles:
- Window selection state
- Entity (path/cluster) selection state
- Cross-component highlighting
- Selection history
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class WindowSelection:
    """Represents a selected window in the network."""
    window_type: str  # 'early', 'middle', 'late', 'custom'
    start_layer: int
    end_layer: int
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class EntitySelection:
    """Represents a selected entity (path or cluster)."""
    entity_type: str  # 'path' or 'cluster'
    entity_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class SelectionManager:
    """Manages selection state across Network Explorer components."""
    
    def __init__(self):
        """Initialize the SelectionManager."""
        self.current_window: Optional[WindowSelection] = None
        self.selected_entities: List[EntitySelection] = []
        self.highlighted_entities: List[str] = []
        self.selection_history: List[Tuple[str, Any]] = []
        self.max_history = 50
        
    def select_window(self, window_type: str, start_layer: int, end_layer: int) -> WindowSelection:
        """Select a window in the network."""
        window = WindowSelection(
            window_type=window_type,
            start_layer=start_layer,
            end_layer=end_layer
        )
        self.current_window = window
        self._add_to_history('window', window)
        return window
    
    def select_entity(self, entity_type: str, entity_id: str, 
                     metadata: Optional[Dict[str, Any]] = None,
                     multi_select: bool = False) -> EntitySelection:
        """Select an entity (path or cluster)."""
        entity = EntitySelection(
            entity_type=entity_type,
            entity_id=entity_id,
            metadata=metadata or {}
        )
        
        if not multi_select:
            self.selected_entities = [entity]
        else:
            # Add to selection if not already selected
            if not any(e.entity_id == entity_id for e in self.selected_entities):
                self.selected_entities.append(entity)
        
        self._add_to_history('entity', entity)
        return entity
    
    def deselect_entity(self, entity_id: str) -> None:
        """Deselect an entity."""
        self.selected_entities = [
            e for e in self.selected_entities if e.entity_id != entity_id
        ]
    
    def clear_selection(self) -> None:
        """Clear all entity selections."""
        self.selected_entities = []
        self.highlighted_entities = []
    
    def highlight_entities(self, entity_ids: List[str]) -> None:
        """Set highlighted entities."""
        self.highlighted_entities = entity_ids
    
    def add_highlight(self, entity_id: str) -> None:
        """Add an entity to highlights."""
        if entity_id not in self.highlighted_entities:
            self.highlighted_entities.append(entity_id)
    
    def remove_highlight(self, entity_id: str) -> None:
        """Remove an entity from highlights."""
        self.highlighted_entities = [
            e for e in self.highlighted_entities if e != entity_id
        ]
    
    def get_selection_state(self) -> Dict[str, Any]:
        """Get the current selection state."""
        return {
            'window': {
                'type': self.current_window.window_type if self.current_window else None,
                'start_layer': self.current_window.start_layer if self.current_window else None,
                'end_layer': self.current_window.end_layer if self.current_window else None
            },
            'selected_entities': [
                {
                    'type': e.entity_type,
                    'id': e.entity_id,
                    'metadata': e.metadata
                }
                for e in self.selected_entities
            ],
            'highlighted_entities': self.highlighted_entities
        }
    
    def restore_selection_state(self, state: Dict[str, Any]) -> None:
        """Restore selection state from a dictionary."""
        # Restore window selection
        if state.get('window') and state['window'].get('type'):
            self.select_window(
                state['window']['type'],
                state['window']['start_layer'],
                state['window']['end_layer']
            )
        
        # Restore entity selections
        self.selected_entities = []
        for entity_data in state.get('selected_entities', []):
            self.select_entity(
                entity_data['type'],
                entity_data['id'],
                entity_data.get('metadata', {}),
                multi_select=True
            )
        
        # Restore highlights
        self.highlighted_entities = state.get('highlighted_entities', [])
    
    def _add_to_history(self, action_type: str, data: Any) -> None:
        """Add an action to the selection history."""
        self.selection_history.append((action_type, data))
        # Trim history if it exceeds max size
        if len(self.selection_history) > self.max_history:
            self.selection_history = self.selection_history[-self.max_history:]
    
    def get_history(self, limit: Optional[int] = None) -> List[Tuple[str, Any]]:
        """Get selection history."""
        if limit:
            return self.selection_history[-limit:]
        return self.selection_history

# Utility functions for Dash callbacks
def parse_window_selection(window_data: Dict[str, Any]) -> Optional[WindowSelection]:
    """Parse window selection from store data."""
    if not window_data or not window_data.get('type'):
        return None
    
    return WindowSelection(
        window_type=window_data['type'],
        start_layer=window_data['start_layer'],
        end_layer=window_data['end_layer']
    )

def parse_entity_selections(entity_data: List[Dict[str, Any]]) -> List[EntitySelection]:
    """Parse entity selections from store data."""
    selections = []
    for item in entity_data:
        selections.append(EntitySelection(
            entity_type=item['type'],
            entity_id=item['id'],
            metadata=item.get('metadata', {})
        ))
    return selections