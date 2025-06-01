#!/usr/bin/env python3
"""
Verify that all Concept MRI components are working correctly.
"""
import sys
from pathlib import Path

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))

def verify_imports():
    """Test that all components can be imported."""
    print("Verifying component imports...")
    errors = []
    
    # Test imports
    imports_to_test = [
        ("dash", "dash"),
        ("dash_bootstrap_components", "dbc"),
        ("dash_uploader", "du"),
        ("concept_mri.config.settings", "settings"),
        ("concept_mri.components.controls.model_upload", "model_upload"),
        ("concept_mri.components.controls.dataset_upload", "dataset_upload"),
        ("concept_mri.components.controls.clustering_panel", "clustering_panel"),
        ("concept_mri.components.controls.api_keys_panel", "api_keys_panel"),
        ("concept_mri.components.visualizations.sankey_wrapper", "sankey_wrapper"),
        ("concept_mri.components.layouts.main_layout", "main_layout"),
        ("concept_mri.core.model_interface", "ModelInterface"),
        ("concept_fragmentation.activation.collector", "ActivationCollector"),
    ]
    
    for module_path, module_name in imports_to_test:
        try:
            exec(f"import {module_path}")
            print(f"✓ Successfully imported {module_path}")
        except ImportError as e:
            errors.append(f"✗ Failed to import {module_path}: {e}")
        except Exception as e:
            errors.append(f"✗ Error importing {module_path}: {e}")
    
    return errors

def verify_component_creation():
    """Test that components can be instantiated."""
    print("\nVerifying component instantiation...")
    errors = []
    
    # Import what we need
    try:
        from concept_mri.components.controls.model_upload import ModelUploadPanel
        from concept_mri.components.controls.dataset_upload import DatasetUploadPanel
        from concept_mri.components.controls.clustering_panel import ClusteringPanel
        from concept_mri.components.controls.api_keys_panel import APIKeysPanel
        from concept_mri.components.visualizations.sankey_wrapper import SankeyWrapper
    except ImportError as e:
        errors.append(f"Failed to import components: {e}")
        return errors
    
    # Test instantiation
    components_to_test = [
        ("ModelUploadPanel", ModelUploadPanel),
        ("DatasetUploadPanel", DatasetUploadPanel),
        ("ClusteringPanel", ClusteringPanel),
        ("APIKeysPanel", APIKeysPanel),
        ("SankeyWrapper", lambda: SankeyWrapper("test-id")),
    ]
    
    for name, component_class in components_to_test:
        try:
            if callable(component_class):
                component = component_class()
            else:
                component = component_class()
            print(f"✓ Successfully created {name}")
            
            # Try to create the component's UI
            if hasattr(component, 'create_component'):
                ui = component.create_component()
                print(f"  ✓ Successfully created UI for {name}")
        except Exception as e:
            errors.append(f"✗ Failed to create {name}: {e}")
    
    return errors

def verify_dependencies():
    """Check for required dependencies."""
    print("\nVerifying dependencies...")
    errors = []
    
    required_packages = [
        'dash',
        'dash_bootstrap_components',
        'plotly',
        'numpy',
        'pandas',
        'torch'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            errors.append(f"✗ Missing required package: {package}")
    
    # Check optional but recommended packages
    optional_packages = ['dash_uploader', 'dash_daq']
    for package in optional_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed (optional)")
        except ImportError:
            print(f"! {package} not installed (optional)")
    
    return errors

def check_directory_structure():
    """Verify required directories exist."""
    print("\nChecking directory structure...")
    errors = []
    
    from concept_mri.config.settings import DATA_DIR, CACHE_DIR, UPLOAD_DIR, EXPORT_DIR
    
    required_dirs = [
        ("Data directory", DATA_DIR),
        ("Cache directory", CACHE_DIR),
        ("Upload directory", UPLOAD_DIR),
        ("Export directory", EXPORT_DIR),
    ]
    
    for name, dir_path in required_dirs:
        if dir_path.exists():
            print(f"✓ {name} exists: {dir_path}")
        else:
            errors.append(f"✗ {name} missing: {dir_path}")
    
    return errors

def main():
    """Run all verification tests."""
    print("=== Concept MRI Component Verification ===\n")
    
    all_errors = []
    
    # Run tests
    all_errors.extend(verify_dependencies())
    all_errors.extend(check_directory_structure())
    all_errors.extend(verify_imports())
    all_errors.extend(verify_component_creation())
    
    # Summary
    print("\n=== Verification Summary ===")
    if all_errors:
        print(f"\n❌ Found {len(all_errors)} errors:\n")
        for error in all_errors:
            print(f"  {error}")
        print("\nPlease fix these errors before proceeding.")
        return 1
    else:
        print("\n✅ All components verified successfully!")
        print("\nNext steps:")
        print("1. Run the app with: python concept_mri/app.py")
        print("2. Test each component in the UI")
        print("3. Proceed with adding window manager")
        return 0

if __name__ == "__main__":
    exit(main())