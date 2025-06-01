#!/usr/bin/env python3
"""
Add wrapper classes to components that need them.
"""
import os
from pathlib import Path

# Components that need wrapper classes
components_to_fix = {
    'dataset_upload.py': 'DatasetUploadPanel',
    'clustering_panel.py': 'ClusteringPanel', 
    'api_keys_panel.py': 'APIKeysPanel'
}

def add_wrapper_class(file_path, class_name):
    """Add a wrapper class to a component file."""
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if class already exists
    if f'class {class_name}' in content:
        print(f"✓ {class_name} already exists in {file_path}")
        return False
    
    # Find the first function definition
    import_end = content.rfind(')')
    if 'from concept_mri.config.settings import' in content:
        # Find the end of the imports
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith(('import', 'from', '#', '"""')) and ')' not in line:
                # Insert the class before the first function
                wrapper_code = f'''

class {class_name}:
    """{class_name.replace('Panel', ' panel')} component."""
    
    def __init__(self):
        """Initialize the {class_name.lower().replace('panel', ' panel')}."""
        self.id_prefix = "{class_name.lower().replace('panel', '').replace('uploadpanel', 'upload')}"
    
    def create_component(self):
        """Create and return the component layout."""
        return create_{class_name.lower().replace('panel', '_panel').replace('uploadpanel', '_upload')}()
'''
                lines.insert(i, wrapper_code)
                content = '\n'.join(lines)
                break
    
    # Write the updated content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Added {class_name} to {file_path}")
    return True

def main():
    """Add wrapper classes to all components."""
    controls_dir = Path(__file__).parent.parent / 'components' / 'controls'
    
    for filename, class_name in components_to_fix.items():
        file_path = controls_dir / filename
        if file_path.exists():
            add_wrapper_class(file_path, class_name)
        else:
            print(f"✗ File not found: {file_path}")

if __name__ == "__main__":
    main()