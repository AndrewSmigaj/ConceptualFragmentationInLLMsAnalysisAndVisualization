#!/usr/bin/env python3
"""Analyze trajectory results."""

import json
from pathlib import Path

def main():
    # Load results
    base_dir = Path(__file__).parent
    results_dir = base_dir / "trajectory_analysis_k3"
    latest_file = sorted(results_dir.glob("trajectory_analysis_*.json"))[-1]
    
    with open(latest_file) as f:
        data = json.load(f)
    
    print("=== TOP TRAJECTORY PATHS (k=3) ===\n")
    
    for window in ['early', 'middle', 'late']:
        print(f"\n{window.upper()} WINDOW:")
        paths = data['window_analysis'][window]['archetypal_paths'][:5]
        
        for i, path in enumerate(paths):
            print(f"\n  Path {i+1}: {path['path'].replace('→', '->')}")
            print(f"  Count: {path['count']} ({path['percentage']:.1f}%)")
            
            # Grammatical analysis
            gram = path['grammatical_analysis']
            print(f"  POS Distribution: {', '.join(f'{k}:{v:.1f}%' for k,v in list(gram['pos_distribution'].items())[:3])}")
            
            # Semantic analysis  
            sem = path['semantic_analysis']
            print(f"  Semantic Categories: {', '.join(f'{k}:{v:.1f}%' for k,v in list(sem['semantic_categories'].items())[:3])}")
            
            # Example tokens
            print(f"  Examples: {', '.join(path['example_tokens'][:5])}")

if __name__ == "__main__":
    main()