"""
Generate path analysis tables for Concept MRI visualization.

Creates detailed tables showing all paths in each window with:
- Complete path sequences
- Frequencies and percentages
- Example words
- Grammatical analysis
- Stability metrics
"""

import json
from pathlib import Path
import sys

# Add project root to path
root_dir = Path(__file__).parent.parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


def create_path_table_html(window_data, window_name):
    """Create HTML table for paths in a window."""
    
    html = f"""
    <div class="window-section">
        <h2>{window_name.capitalize()} Window ({window_data['window_info']['layer_range']})</h2>
        <div class="window-stats">
            <span class="stat">Total Paths: {window_data['summary']['total_unique_paths']}</span>
            <span class="stat">Common Paths (≥10 words): {window_data['summary']['common_paths_gte10']}</span>
            <span class="stat">Rare Paths (<3 words): {window_data['summary']['rare_paths_lt3']}</span>
            <span class="stat">Singleton Paths: {window_data['summary']['singleton_paths']}</span>
        </div>
        
        <table class="path-table">
            <thead>
                <tr>
                    <th width="5%">#</th>
                    <th width="25%">Path Sequence</th>
                    <th width="15%">Frequency</th>
                    <th width="15%">Grammar</th>
                    <th width="30%">Example Words</th>
                    <th width="10%">Stability</th>
                </tr>
            </thead>
            <tbody>
    """
    
    # Add rows for each path
    for idx, path in enumerate(window_data['all_paths']):
        # Determine row class based on frequency
        if path['frequency'] >= 10:
            row_class = "common-path"
        elif path['frequency'] >= 3:
            row_class = "moderate-path"
        elif path['frequency'] == 1:
            row_class = "singleton-path"
        else:
            row_class = "rare-path"
        
        # Format path sequence
        path_seq = " → ".join(path['cluster_sequence'])
        
        # Format frequency
        freq_text = f"{path['frequency']} ({path['percentage']:.1f}%)"
        
        # Get grammatical info
        gram_info = path.get('category_analysis', {})
        dominant_gram = gram_info.get('dominant_grammatical', 'unknown')
        gram_color = gram_info.get('dominant_color', '#7f7f7f')
        
        # Format examples
        examples = ", ".join(path['example_words'][:8])
        if len(path['example_words']) > 8:
            examples += "..."
        
        # Add "all words" for rare paths
        if path['frequency'] <= 5 and 'all_words' in path:
            examples = "<b>All:</b> " + ", ".join(path['all_words'])
        
        # Stability info
        stability = path['characteristics']['stability']
        stability_class = "stable" if stability > 0.7 else "unstable" if stability < 0.3 else "moderate"
        stability_text = f"{stability:.1%}"
        
        html += f"""
                <tr class="{row_class}">
                    <td class="rank">{idx + 1}</td>
                    <td class="path-sequence">{path_seq}</td>
                    <td class="frequency">{freq_text}</td>
                    <td class="grammar">
                        <span class="gram-badge" style="background-color: {gram_color}">{dominant_gram}</span>
                    </td>
                    <td class="examples">{examples}</td>
                    <td class="stability {stability_class}">{stability_text}</td>
                </tr>
        """
    
    html += """
            </tbody>
        </table>
    </div>
    """
    
    return html


def create_transition_analysis_html(transition_data):
    """Create HTML for transition analysis between windows."""
    
    html = """
    <div class="transition-section">
        <h2>Cross-Window Transitions</h2>
    """
    
    for trans_type, trans_info in transition_data.items():
        trans_title = trans_type.replace('_', ' ').title()
        
        html += f"""
        <div class="transition-block">
            <h3>{trans_title}</h3>
            <div class="transition-stats">
                <span>Path Convergence: {trans_info['from_paths_count']} → {trans_info['to_paths_count']} paths</span>
                <span>Convergence Ratio: {trans_info['convergence_ratio']:.2f}</span>
            </div>
            <table class="transition-table">
                <thead>
                    <tr>
                        <th>From Path</th>
                        <th>To Path</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Show top transitions
        for trans in trans_info['top_transitions'][:5]:
            from_path = " → ".join(trans['from_sequence'])
            to_path = " → ".join(trans['to_sequence'])
            
            html += f"""
                    <tr>
                        <td class="path-sequence">{from_path}</td>
                        <td class="path-sequence">{to_path}</td>
                        <td>{trans['count']}</td>
                        <td>{trans['percentage']:.1f}%</td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        """
    
    html += "</div>"
    return html


def generate_path_analysis_page(results_dir):
    """Generate complete path analysis HTML page."""
    
    # Load MRI data
    mri_data_path = Path(results_dir) / "concept_mri_data.json"
    with open(mri_data_path, 'r') as f:
        mri_data = json.load(f)
    
    # Create HTML page
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>GPT-2 Concept MRI: Path Analysis</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        h1 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .subtitle {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
        }
        
        .window-section {
            margin-bottom: 40px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
        }
        
        .window-section h2 {
            color: #34495e;
            margin-top: 0;
        }
        
        .window-stats {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        
        .stat {
            background: #ecf0f1;
            padding: 8px 16px;
            border-radius: 4px;
            font-size: 14px;
        }
        
        .path-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }
        
        .path-table th {
            background: #34495e;
            color: white;
            padding: 12px 8px;
            text-align: left;
            position: sticky;
            top: 0;
        }
        
        .path-table td {
            padding: 10px 8px;
            border-bottom: 1px solid #ecf0f1;
        }
        
        .path-table tr:hover {
            background: #f8f9fa;
        }
        
        .rank {
            font-weight: bold;
            color: #7f8c8d;
        }
        
        .path-sequence {
            font-family: 'Courier New', monospace;
            font-size: 12px;
            color: #2c3e50;
        }
        
        .frequency {
            font-weight: bold;
        }
        
        .gram-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            color: white;
            font-size: 12px;
            font-weight: bold;
        }
        
        .examples {
            font-size: 13px;
            color: #555;
        }
        
        .stability {
            text-align: center;
            font-weight: bold;
        }
        
        .stability.stable { color: #27ae60; }
        .stability.moderate { color: #f39c12; }
        .stability.unstable { color: #e74c3c; }
        
        .common-path { background: #e8f8f5; }
        .moderate-path { background: #fef9e7; }
        .rare-path { background: #fadbd8; }
        .singleton-path { background: #f4ecf7; }
        
        .transition-section {
            margin-top: 40px;
            border-top: 2px solid #34495e;
            padding-top: 30px;
        }
        
        .transition-block {
            margin-bottom: 30px;
        }
        
        .transition-stats {
            display: flex;
            gap: 30px;
            margin-bottom: 15px;
            font-size: 14px;
            color: #555;
        }
        
        .transition-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }
        
        .transition-table th {
            background: #7f8c8d;
            color: white;
            padding: 10px;
            text-align: left;
        }
        
        .transition-table td {
            padding: 8px 10px;
            border-bottom: 1px solid #ecf0f1;
        }
        
        .key-insight {
            background: #3498db;
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            margin: 20px 0;
            font-size: 16px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 GPT-2 Concept MRI: Complete Path Analysis</h1>
        <p class="subtitle">Tracking how 566 words flow through GPT-2's 12 layers</p>
        
        <div class="key-insight">
            📊 Key Finding: Massive convergence from {early_paths} → {middle_paths} → {late_paths} unique paths
        </div>
    """
    
    # Add path tables for each window
    for window in ['early', 'middle', 'late']:
        if window in mri_data['paths']:
            html += create_path_table_html(mri_data['paths'][window], window)
    
    # Add transition analysis
    if 'transitions' in mri_data:
        html += create_transition_analysis_html(mri_data['transitions'])
    
    # Add key patterns summary
    html += """
        <div class="key-patterns">
            <h2>Key Patterns</h2>
            <ul>
                <li><strong>Grammatical Organization:</strong> Paths are organized by grammatical function (noun/adjective/adverb) rather than semantic meaning</li>
                <li><strong>Dominant Path:</strong> 72.8% of all words converge to a single path in middle layers</li>
                <li><strong>Stability Shift:</strong> High stability in early layers (72.4%) drops to low stability in middle/late layers (~34%)</li>
                <li><strong>Semantic Blindness:</strong> Animals, objects, and abstract concepts all follow the same paths</li>
            </ul>
        </div>
    """
    
    # Replace placeholders
    early_paths = mri_data['paths']['early']['summary']['total_unique_paths']
    middle_paths = mri_data['paths']['middle']['summary']['total_unique_paths']
    late_paths = mri_data['paths']['late']['summary']['total_unique_paths']
    
    html = html.replace('{early_paths}', str(early_paths))
    html = html.replace('{middle_paths}', str(middle_paths))
    html = html.replace('{late_paths}', str(late_paths))
    
    html += """
    </div>
</body>
</html>
    """
    
    # Save HTML file
    output_path = Path(results_dir) / "path_analysis_tables.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"Generated path analysis tables: {output_path}")
    return output_path


def main(results_dir=None):
    """Main function to generate path analysis tables."""
    
    if results_dir is None:
        results_dir = Path("results/unified_cta_config/unified_cta_20250524_073316")
    else:
        results_dir = Path(results_dir)
    
    print("Generating path analysis tables...")
    output_path = generate_path_analysis_page(results_dir)
    print(f"\nPath analysis tables generated successfully!")
    
    return output_path


if __name__ == "__main__":
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = None
    
    main(results_dir)