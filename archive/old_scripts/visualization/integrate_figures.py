#!/usr/bin/env python3
"""
Integrate figures into the foundations.md paper.

This script:
1. Creates a copy of foundations.md with figures integrated
2. Inserts figure references at appropriate sections based on content
3. Adds figure captions that match the paper's content
"""

import os
import re
import shutil
from datetime import datetime

# Set up paths
repo_root = os.path.dirname(os.path.abspath(__file__))
foundations_path = os.path.join(repo_root, "concept_fragmentation", "foundations.md")
figures_dir = os.path.join(repo_root, "figures")
output_path = os.path.join(repo_root, "concept_fragmentation", "foundations_with_figures.md")

# Define figure insertion points
figure_insertions = [
    {
        "figure": "optimal_clusters.png",
        "section_pattern": r"4 Statistical Robustness of Cluster Structures",
        "caption": "**Figure 1: Optimal number of clusters (k\\*) by layer.** The number of natural clusters peaks in the middle layers before consolidating in the final layers, suggesting the model discovers complex feature interactions before distilling them into decision-relevant representations.",
        "position": "after",
        "offset": 0
    },
    {
        "figure": "intra_class_distance.png",
        "section_pattern": r"4\.1 Path Reproducibility Across Seeds",
        "caption": "**Figure 2: Intra-class pairwise distance (ICPD) by layer.** ICPD increases across layers, indicating greater within-class differentiation as the network forms specialized subgroups.",
        "position": "after",
        "offset": 0
    },
    {
        "figure": "subspace_angle.png",
        "section_pattern": r"4\.2 Trajectory Coherence",
        "caption": "**Figure 3: Subspace angles between class representations by layer.** Angles decrease as the network learns to better separate classes in latent space.",
        "position": "after",
        "offset": 0
    },
    {
        "figure": "cluster_entropy.png",
        "section_pattern": r"4\.3 Feature Attribution for Cluster Transitions",
        "caption": "**Figure 4: Cluster entropy by layer.** Entropy decreases as representations become more class-specific, indicating the network is learning coherent class boundaries.",
        "position": "after",
        "offset": 0
    },
    {
        "figure": "trajectory_basic.png",
        "section_pattern": r"5\.1 Stepped-Layer Trajectory Visualization",
        "caption": "**Figure 5: UMAP projection of activation trajectories.** Paths through the network's activation space show distinct patterns for different classes, with survivor paths (orange) following more consistent trajectories than non-survivor paths (blue).",
        "position": "after",
        "offset": 0  # Insert first
    },
    {
        "figure": "trajectory_annotated.png",
        "section_pattern": r"5\.1 Stepped-Layer Trajectory Visualization",
        "caption": "**Figure 6: Annotated activation trajectories with LLM-derived explanations.** The three main archetypes are highlighted: privileged path (purple, 0→2→0), disadvantaged path (brown, 1→1→1), and ambiguous path (pink, 2→0→1).",
        "position": "after",
        "offset": 1  # Insert second
    },
    {
        "figure": "trajectory_by_endpoint_cluster.png",
        "section_pattern": r"5\.1 Stepped-Layer Trajectory Visualization",
        "caption": "**Figure 7: Layer 3 clustering.** Final layer representations show clear separation between clusters with differing survival rates, demonstrating the model's learned decision boundaries.",
        "position": "after",
        "offset": 2  # Insert third
    }
]

def integrate_figures():
    """Insert figures into the paper at appropriate locations."""
    print(f"Reading foundations.md from {foundations_path}")
    
    # Check if files exist
    if not os.path.exists(foundations_path):
        print(f"Error: Could not find {foundations_path}")
        return False
    
    # Make backup
    backup_path = f"{foundations_path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(foundations_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    # Read the paper content
    with open(foundations_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Process each figure insertion
    modified_content = content
    for insertion in figure_insertions:
        figure_path = os.path.join("figures", insertion["figure"])
        if not os.path.exists(os.path.join(repo_root, figure_path)):
            print(f"Warning: Figure {figure_path} does not exist. Skipping.")
            continue
        
        # Create figure markdown
        figure_md = f"\n\n![{insertion['figure']}]({figure_path})\n\n{insertion['caption']}\n"
        
        # Find insertion point
        match = re.search(insertion["section_pattern"], modified_content)
        if not match:
            print(f"Warning: Could not find section matching '{insertion['section_pattern']}'. Skipping.")
            continue
        
        if insertion["position"] == "after":
            # Find the end of the section header line
            end_of_line = modified_content.find('\n', match.end())
            if end_of_line != -1:
                # Insert after the section header
                modified_content = modified_content[:end_of_line] + figure_md + modified_content[end_of_line:]
                print(f"Inserted figure {insertion['figure']} after section matching '{insertion['section_pattern']}'")
            else:
                print(f"Warning: Could not find end of line for section matching '{insertion['section_pattern']}'. Skipping.")
        elif insertion["position"] == "before":
            # Insert before the section header
            modified_content = modified_content[:match.start()] + figure_md + modified_content[match.start():]
            print(f"Inserted figure {insertion['figure']} before section matching '{insertion['section_pattern']}'")
    
    # Write the modified content
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print(f"Successfully created {output_path} with integrated figures")
    return True

def create_visualization_section():
    """Create a visualization section in case one doesn't exist."""
    # This is a fallback if we need to add a visualization section from scratch
    visualization_section = """
5 Visualizations

We present several visualizations to illustrate the concepts and findings discussed in this paper.

5.1 Fragmentation Metrics

The following figures show key fragmentation metrics across network layers:

![optimal_clusters.png](figures/optimal_clusters.png)

**Figure 1: Optimal number of clusters (k*) by layer.** The number of natural clusters peaks in the middle layers before consolidating in the final layers, suggesting the model discovers complex feature interactions before distilling them into decision-relevant representations.

![intra_class_distance.png](figures/intra_class_distance.png)

**Figure 2: Intra-class pairwise distance (ICPD) by layer.** ICPD increases across layers, indicating greater within-class differentiation as the network forms specialized subgroups.

![subspace_angle.png](figures/subspace_angle.png)

**Figure 3: Subspace angles between class representations by layer.** Angles decrease as the network learns to better separate classes in latent space.

![cluster_entropy.png](figures/cluster_entropy.png)

**Figure 4: Cluster entropy by layer.** Entropy decreases as representations become more class-specific, indicating the network is learning coherent class boundaries.

5.2 Trajectory Visualizations

The following visualizations show how data points move through the network's activation space:

![trajectory_basic.png](figures/trajectory_basic.png)

**Figure 5: UMAP projection of activation trajectories.** Paths through the network's activation space show distinct patterns for different classes, with survivor paths (orange) following more consistent trajectories than non-survivor paths (blue).

![trajectory_annotated.png](figures/trajectory_annotated.png)

**Figure 6: Annotated activation trajectories with LLM-derived explanations.** The three main archetypes are highlighted: privileged path (purple, 0→2→0), disadvantaged path (brown, 1→1→1), and ambiguous path (pink, 2→0→1).

![trajectory_by_endpoint_cluster.png](figures/trajectory_by_endpoint_cluster.png)

**Figure 7: Layer 3 clustering.** Final layer representations show clear separation between clusters with differing survival rates, demonstrating the model's learned decision boundaries.
"""
    
    # Check if sections 5 or 5.1 already exist
    with open(foundations_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if "5 Visualizations" not in content and "5.1 Stepped-Layer Trajectory Visualization" not in content:
        print("No visualization section found. Creating one...")
        
        # Find section 4 to insert after it
        match = re.search(r'4 Statistical Robustness of Cluster Structures', content)
        if match:
            # Find the end of section 4
            next_section_match = re.search(r'\n[56] [A-Z]', content[match.end():])
            if next_section_match:
                insertion_point = match.end() + next_section_match.start()
                
                modified_content = content[:insertion_point] + visualization_section + content[insertion_point:]
                
                # Write the modified content
                backup_path = f"{foundations_path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}.nosection"
                shutil.copy2(foundations_path, backup_path)
                
                output_path_with_section = f"{foundations_path}.with_vis_section"
                with open(output_path_with_section, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                
                print(f"Created file with visualization section at {output_path_with_section}")
                return True
    
    return False

if __name__ == "__main__":
    success = integrate_figures()
    if not success:
        print("Trying to create a visualization section...")
        create_visualization_section()
    
    print("\nDone. Please review the changes in foundations_with_figures.md")