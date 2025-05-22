"""
Helper script to extract LLM analysis results and format them for LaTeX.
"""

import json
import sys
import pathlib
import textwrap
import pandas as pd
from typing import Dict, Any

def latex_escape(s: str) -> str:
    """Escape special characters for LaTeX."""
    return (s.replace('_', '\\_')
            .replace('%', '\\%')
            .replace('&', '\\&')
            .replace('#', '\\#'))

def format_cluster_labels(data: Dict[str, Any], dataset: str) -> str:
    """Format cluster labels as LaTeX itemize environment."""
    labels = data["cluster_labels"]
    
    # Group by layer
    by_layer = {}
    for cluster_id, label in labels.items():
        layer = cluster_id.split('C')[0]
        if layer not in by_layer:
            by_layer[layer] = []
        by_layer[layer].append((cluster_id, label))
    
    # Format as LaTeX
    lines = [f"\\subsection{{Cluster Labels for {dataset.title()} Dataset}}"]
    lines.append("\\begin{itemize}")
    
    for layer in sorted(by_layer.keys()):
        lines.append(f"\\item \\textbf{{{layer}}}:")
        inner_lines = []
        inner_lines.append("\\begin{itemize}")
        for cluster_id, label in sorted(by_layer[layer]):
            if label.lower().startswith("data not provided") or label.lower().startswith("unknown"):
                continue
            inner_lines.append(f"\\item {cluster_id}: {latex_escape(label)}")
        inner_lines.append("\\end{itemize}")
        # only add if we kept at least one label
        if len(inner_lines) > 2:
            lines.extend(inner_lines)
    
    lines.append("\\end{itemize}")
    return "\n".join(lines)

def format_path_narratives(data: Dict[str, Any], dataset: str) -> str:
    """Format path narratives as LaTeX quotes."""
    narratives = data["path_narratives"]
    
    lines = [f"\\subsection{{Path Narratives for {dataset.title()} Dataset}}"]
    
    # Take first 3 narratives as examples
    for i, (path_id, narrative) in enumerate(list(narratives.items())[:3]):
        wrapped = textwrap.fill(narrative, width=80)
        lines.extend([
            f"\\textbf{{Path {path_id}}}:",
            "\\begin{quote}",
            latex_escape(wrapped),
            "\\end{quote}",
            ""
        ])
    
    return "\n".join(lines)

def format_bias_metrics(data: Dict[str, Any], dataset: str) -> str:
    """Format bias metrics as LaTeX table."""
    if "bias_report" not in data or not data["bias_report"]:
        return f"% No bias metrics available for {dataset}"
        
    bias = data["bias_report"]
    if not isinstance(bias, dict) or "average_bias_scores" not in bias.get("summary", {}):
        return f"% Invalid bias report format for {dataset}"
    
    lines = [f"\\subsection{{Bias Metrics for {dataset.title()} Dataset}}"]
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{lr}")
    lines.append("\\toprule")
    lines.append("Demographic Factor & Average Bias Score \\\\")
    lines.append("\\midrule")
    
    for factor, score in bias["summary"]["average_bias_scores"].items():
        lines.append(f"{factor.title()} & {score:.4f} \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{Bias scores for {dataset.title()} dataset paths}}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)

def main():
    """Extract results and write LaTeX fragments."""
    root = pathlib.Path(__file__).resolve().parents[1]
    
    for dataset in ["titanic", "heart"]:
        # Load LLM analysis results
        results_file = root / "results" / "llm" / f"{dataset}_seed_0_analysis.json"
        if not results_file.exists():
            print(f"Warning: No results file found for {dataset}")
            continue
            
        with open(results_file) as f:
            data = json.load(f)
        
        # Create output directory
        out_dir = root / "arxiv_submission" / "sections" / "generated"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Write cluster labels
        labels_file = out_dir / f"{dataset}_labels.tex"
        with open(labels_file, 'w', encoding='utf-8') as f:
            f.write(format_cluster_labels(data, dataset))
        print(f"Wrote cluster labels to {labels_file}")
        
        # Write path narratives
        narratives_file = out_dir / f"{dataset}_narratives.tex"
        with open(narratives_file, 'w', encoding='utf-8') as f:
            f.write(format_path_narratives(data, dataset))
        print(f"Wrote path narratives to {narratives_file}")
        
        # Write bias metrics
        bias_file = out_dir / f"{dataset}_bias.tex"
        with open(bias_file, 'w', encoding='utf-8') as f:
            f.write(format_bias_metrics(data, dataset))
        print(f"Wrote bias metrics to {bias_file}")

        # If full_report_latex exists write it too
        if "full_report_latex" in data:
            report_file = out_dir / f"{dataset}_report.tex"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(data["full_report_latex"])
            print(f"Wrote full report to {report_file}")
        elif "full_report" in data:
            # For backward compatibility with old files
            report_file = out_dir / f"{dataset}_report.tex"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(data["full_report"])
            print(f"Wrote legacy full report to {report_file}")

if __name__ == "__main__":
    main() 