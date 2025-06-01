#!/usr/bin/env python3
"""
Fix relative imports in Concept MRI components.
"""
import os
import re
from pathlib import Path

def fix_imports_in_file(file_path):
    """Fix imports in a single file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix imports
    patterns = [
        (r'from config\.settings import', 'from concept_mri.config.settings import'),
        (r'from components\.', 'from concept_mri.components.'),
        (r'from core\.', 'from concept_mri.core.'),
        (r'import config\.', 'import concept_mri.config.'),
        (r'import components\.', 'import concept_mri.components.'),
        (r'import core\.', 'import concept_mri.core.'),
    ]
    
    modified = False
    for pattern, replacement in patterns:
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            modified = True
            content = new_content
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Fixed imports in {file_path}")
        return True
    return False

def main():
    """Fix all imports in Concept MRI."""
    concept_mri_dir = Path(__file__).parent.parent
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk(concept_mri_dir):
        # Skip test and cache directories
        if 'test' in root or '__pycache__' in root or 'venv' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"Found {len(python_files)} Python files to check")
    
    fixed_count = 0
    for file_path in python_files:
        if fix_imports_in_file(file_path):
            fixed_count += 1
    
    print(f"\nFixed imports in {fixed_count} files")

if __name__ == "__main__":
    main()