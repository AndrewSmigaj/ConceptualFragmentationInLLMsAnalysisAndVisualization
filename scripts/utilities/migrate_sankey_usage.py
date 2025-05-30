#!/usr/bin/env python3
"""Migration script to update existing code to use unified SankeyGenerator.

This script helps migrate from the various experimental Sankey implementations
to the new unified SankeyGenerator in concept_fragmentation.visualization.sankey.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple
import argparse


# Mapping of old implementations to new
OLD_IMPORTS = [
    r"from generate_sankey_diagrams import SankeyGenerator",
    r"from generate_k10_sankeys import .*",
    r"from generate_colored_sankeys_k10 import .*",
    r"from generate_enhanced_sankeys_k10 import .*",
    r"from generate_fixed_sankeys_k10 import .*",
]

NEW_IMPORT = "from concept_fragmentation.visualization.sankey import SankeyGenerator"
NEW_CONFIG_IMPORT = "from concept_fragmentation.visualization.configs import SankeyConfig"


def find_files_to_migrate(root_dir: Path, extensions: List[str] = ['.py']) -> List[Path]:
    """Find all Python files that might need migration."""
    files = []
    for ext in extensions:
        files.extend(root_dir.rglob(f'*{ext}'))
    
    # Exclude the new implementation and tests
    excluded_paths = [
        'concept_fragmentation/visualization/sankey.py',
        'concept_fragmentation/visualization/tests/test_sankey.py',
        'scripts/utilities/migrate_sankey_usage.py',
    ]
    
    return [f for f in files if not any(str(f).endswith(ex) for ex in excluded_paths)]


def check_file_needs_migration(file_path: Path) -> bool:
    """Check if a file needs migration."""
    content = file_path.read_text(encoding='utf-8')
    
    # Check for old imports
    for pattern in OLD_IMPORTS:
        if re.search(pattern, content):
            return True
    
    # Check for direct usage of old files
    old_file_patterns = [
        r'generate_sankey_diagrams',
        r'generate_k10_sankeys',
        r'generate_colored_sankeys_k10',
        r'generate_enhanced_sankeys_k10',
        r'generate_fixed_sankeys_k10',
    ]
    
    for pattern in old_file_patterns:
        if re.search(pattern, content):
            return True
    
    return False


def migrate_imports(content: str) -> Tuple[str, List[str]]:
    """Migrate import statements."""
    changes = []
    
    # Replace old imports
    for old_import in OLD_IMPORTS:
        if re.search(old_import, content):
            content = re.sub(old_import, NEW_IMPORT, content)
            changes.append(f"Updated import: {old_import} -> {NEW_IMPORT}")
    
    # Add config import if needed
    if 'SankeyGenerator' in content and NEW_CONFIG_IMPORT not in content:
        # Add after the SankeyGenerator import
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if NEW_IMPORT in line:
                lines.insert(i + 1, NEW_CONFIG_IMPORT)
                changes.append(f"Added config import: {NEW_CONFIG_IMPORT}")
                break
        content = '\n'.join(lines)
    
    return content, changes


def migrate_usage_patterns(content: str) -> Tuple[str, List[str]]:
    """Migrate common usage patterns."""
    changes = []
    
    # Pattern 1: Direct instantiation without config
    pattern = r'generator\s*=\s*SankeyGenerator\(\s*\)'
    if re.search(pattern, content):
        replacement = 'generator = SankeyGenerator()'
        content = re.sub(pattern, replacement, content)
        changes.append("Updated SankeyGenerator instantiation")
    
    # Pattern 2: Using k parameter
    pattern = r'SankeyGenerator\(k=(\d+)\)'
    matches = re.findall(pattern, content)
    for match in matches:
        old = f'SankeyGenerator(k={match})'
        new = f'SankeyGenerator(SankeyConfig(top_n_paths={match}))'
        content = content.replace(old, new)
        changes.append(f"Updated k parameter to config: k={match} -> top_n_paths={match}")
    
    # Pattern 3: generate_sankey method calls
    pattern = r'\.generate_sankey\('
    if re.search(pattern, content):
        content = re.sub(pattern, '.create_figure(', content)
        changes.append("Updated method: generate_sankey -> create_figure")
    
    # Pattern 4: save_sankey method calls
    pattern = r'\.save_sankey\('
    if re.search(pattern, content):
        content = re.sub(pattern, '.save_figure(', content)
        changes.append("Updated method: save_sankey -> save_figure")
    
    return content, changes


def migrate_data_structure(content: str) -> Tuple[str, List[str]]:
    """Migrate data structure if needed."""
    changes = []
    
    # Check if using old data structure directly
    if 'windowed_analysis' in content and 'create_figure' in content:
        # Add comment about data structure
        pattern = r'create_figure\(windowed_analysis'
        if re.search(pattern, content):
            comment = """
# Note: The new SankeyGenerator expects data in this format:
# data = {
#     'windowed_analysis': windowed_analysis,
#     'labels': semantic_labels,  # Optional
#     'purity_data': purity_scores  # Optional
# }
"""
            # Find a good place to insert the comment
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'create_figure' in line and 'windowed_analysis' in line:
                    lines.insert(i, comment)
                    changes.append("Added data structure documentation")
                    break
            content = '\n'.join(lines)
    
    return content, changes


def migrate_file(file_path: Path, dry_run: bool = False) -> List[str]:
    """Migrate a single file."""
    print(f"\nProcessing: {file_path}")
    
    content = file_path.read_text(encoding='utf-8')
    original_content = content
    all_changes = []
    
    # Apply migrations
    content, changes = migrate_imports(content)
    all_changes.extend(changes)
    
    content, changes = migrate_usage_patterns(content)
    all_changes.extend(changes)
    
    content, changes = migrate_data_structure(content)
    all_changes.extend(changes)
    
    # Write changes if not dry run
    if all_changes and not dry_run:
        # Create backup
        backup_path = file_path.with_suffix('.py.bak')
        backup_path.write_text(original_content, encoding='utf-8')
        print(f"  Created backup: {backup_path}")
        
        # Write updated content
        file_path.write_text(content, encoding='utf-8')
        print(f"  Updated file with {len(all_changes)} changes")
    
    # Report changes
    for change in all_changes:
        print(f"  - {change}")
    
    if not all_changes:
        print("  No changes needed")
    
    return all_changes


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(
        description="Migrate code to use unified SankeyGenerator"
    )
    parser.add_argument(
        '--root',
        type=Path,
        default=Path.cwd(),
        help='Root directory to search for files (default: current directory)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without modifying files'
    )
    parser.add_argument(
        '--file',
        type=Path,
        help='Migrate a specific file instead of searching'
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified\n")
    
    if args.file:
        # Migrate single file
        if args.file.exists():
            migrate_file(args.file, dry_run=args.dry_run)
        else:
            print(f"Error: File not found: {args.file}")
    else:
        # Find and migrate all files
        files = find_files_to_migrate(args.root)
        files_needing_migration = []
        
        print(f"Scanning {len(files)} files...")
        for file_path in files:
            if check_file_needs_migration(file_path):
                files_needing_migration.append(file_path)
        
        if not files_needing_migration:
            print("\nNo files need migration!")
            return
        
        print(f"\nFound {len(files_needing_migration)} files that need migration:")
        for f in files_needing_migration:
            print(f"  - {f}")
        
        if not args.dry_run:
            response = input("\nProceed with migration? (y/N): ")
            if response.lower() != 'y':
                print("Migration cancelled")
                return
        
        # Migrate files
        total_changes = 0
        for file_path in files_needing_migration:
            changes = migrate_file(file_path, dry_run=args.dry_run)
            total_changes += len(changes)
        
        print(f"\nMigration complete! Total changes: {total_changes}")
        
        if not args.dry_run:
            print("\nNext steps:")
            print("1. Review the changes in your code")
            print("2. Update any data structures to match the new format")
            print("3. Run your tests to ensure everything works")
            print("4. Delete backup files (*.py.bak) when satisfied")


if __name__ == '__main__':
    main()