#!/usr/bin/env python3
"""Run k=5 clustering using the existing k3 script."""

from pathlib import Path
from cluster_top_10k_k3 import FixedKClusterer

def main():
    base_dir = Path(__file__).parent
    # Use k=5 instead of default k=3
    clusterer = FixedKClusterer(base_dir, k=5)
    clusterer.run_clustering()

if __name__ == "__main__":
    main()