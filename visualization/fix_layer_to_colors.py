#!/usr/bin/env python
"""
Fix script for layer_to_colors initialization in traj_plot.py
"""

import re

def fix_traj_plot():
    # Read the file
    with open('traj_plot.py', 'r') as f:
        content = f.read()
    
    # Find the line with "# Class colors or cluster colors logic" and fix it
    pattern = r'highlight_set = set\(s for s in current_highlight_indices if s in current_samples_to_plot\)\s+\n\s+# Class colors or cluster colors logic.*?point_colors_array = None'
    replacement = 'highlight_set = set(s for s in current_highlight_indices if s in current_samples_to_plot)\n\n        # Class colors or cluster colors logic\n        point_colors_array = None\n        layer_to_colors = {}  # Initialize here to ensure it\'s available throughout'
    
    # Replace using regex
    fixed_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Write the file back
    with open('traj_plot.py', 'w') as f:
        f.write(fixed_content)
    
    print("Successfully fixed layer_to_colors initialization in traj_plot.py")

if __name__ == "__main__":
    fix_traj_plot() 