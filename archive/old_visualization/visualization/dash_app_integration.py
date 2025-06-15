"""
Integration module for adding GPT-2 token path visualization to the dashboard.

This file provides a function to integrate the GPT-2 token path tab into the
main Dash application without requiring changes to the original dash_app.py file.
"""

import os
import sys
from typing import Dict, Any, Optional

# Ensure parent directory is in path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import GPT-2 token tab
from visualization.gpt2_token_tab import create_gpt2_token_tab, register_gpt2_token_callbacks


def integrate_gpt2_tab(app):
    """
    Integrate the GPT-2 token path tab into the main Dash application.
    
    Args:
        app: The Dash application instance
    """
    # Register callbacks
    register_gpt2_token_callbacks(app)


def get_gpt2_tab():
    """
    Get the GPT-2 token path tab component.
    
    Returns:
        Dash Tab component for GPT-2 token path visualization
    """
    return create_gpt2_token_tab()