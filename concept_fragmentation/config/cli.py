"""
Command-line interface for Concept Fragmentation configuration management.

This module provides a command-line interface for viewing, exporting, importing,
validating, and setting configuration values.
"""

import argparse
import sys
import os
import json
import yaml
import pprint
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from .config_manager import ConfigManager
from .config_classes import Config


def setup_logging():
    """Set up logging for the CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def view_command(args):
    """
    View the current configuration.
    
    Args:
        args: Command-line arguments
    """
    cm = ConfigManager()
    config = cm.get_config()
    
    if args.section:
        # View a specific section
        sections = args.section.split('.')
        current = config.to_dict()
        
        for section in sections:
            if section in current:
                current = current[section]
            else:
                print(f"Error: Section not found: {args.section}")
                return 1
        
        if args.format.lower() == 'json':
            output = json.dumps(current, indent=2, sort_keys=True)
        elif args.format.lower() == 'yaml':
            output = yaml.dump(current, default_flow_style=False)
        else:  # python
            output = pprint.pformat(current, indent=2)
    else:
        # View the entire configuration
        if args.format.lower() == 'json':
            output = config.to_json()
        elif args.format.lower() == 'yaml':
            output = config.to_yaml()
        else:  # python
            output = pprint.pformat(config.to_dict(), indent=2)
    
    # Write to file or print to stdout
    if args.output:
        # Create parent directory if it doesn't exist
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Configuration written to {args.output}")
    else:
        print(output)
        
    return 0


def export_command(args):
    """
    Export configuration to a file.
    
    Args:
        args: Command-line arguments
    """
    cm = ConfigManager()
    
    # Get the appropriate configuration
    if args.dataset and args.experiment:
        config = cm.get_experiment_config(args.dataset, args.experiment)
    else:
        config = cm.get_config()
    
    # Create parent directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export to the specified format
    if args.format.lower() == 'json':
        config.to_json(args.output)
    else:  # yaml
        config.to_yaml(args.output)
    
    print(f"Configuration exported to {args.output}")
    return 0


def import_command(args):
    """
    Import configuration from a file.
    
    Args:
        args: Command-line arguments
    """
    try:
        cm = ConfigManager()
        
        # Load configuration from file
        config = Config.from_file(args.input)
        
        # Merge with existing configuration if requested
        if args.merge:
            current_config = cm.get_config()
            config = current_config.merge(config)
        
        # Validate if requested
        if args.validate:
            # Set the config temporarily
            temp_cm = ConfigManager()
            temp_cm.set_config(config)
            
            if temp_cm.validate():
                print("Configuration validation successful.")
            else:
                print("Configuration validation failed.")
                return 1
        
        # Set as current configuration
        cm.set_config(config)
        print(f"Configuration imported from {args.input}")
        
        # Save to default location if requested
        if args.save:
            default_path = os.path.join(os.getcwd(), "config.json")
            cm.save_config(default_path)
            print(f"Configuration saved to {default_path}")
        
        return 0
        
    except Exception as e:
        print(f"Error importing configuration: {e}")
        return 1


def validate_command(args):
    """
    Validate a configuration file.
    
    Args:
        args: Command-line arguments
    """
    try:
        # Create a ConfigManager for validation
        cm = ConfigManager()
        
        if args.input:
            # Load configuration from file
            config = Config.from_file(args.input)
            
            # Set the config and validate
            cm.set_config(config)
            print(f"Validating configuration file: {args.input}")
        else:
            # Validate the current configuration
            print("Validating current configuration")
        
        if cm.validate():
            print("Configuration validation successful.")
            return 0
        else:
            print("Configuration validation failed.")
            return 1
            
    except Exception as e:
        print(f"Error validating configuration: {e}")
        return 1


def set_command(args):
    """
    Set a configuration value.
    
    Args:
        args: Command-line arguments
    """
    try:
        cm = ConfigManager()
        config = cm.get_config()
        
        # Parse the value based on type
        if args.value.lower() == 'true':
            value = True
        elif args.value.lower() == 'false':
            value = False
        else:
            try:
                # Try to parse as number
                if '.' in args.value:
                    value = float(args.value)
                else:
                    value = int(args.value)
            except ValueError:
                # Treat as string
                value = args.value
        
        # Update the configuration
        updated_config = config.update(**{args.key: value})
        cm.set_config(updated_config)
        
        print(f"Set {args.key} = {value}")
        
        # Save to file if requested
        if args.save:
            default_path = os.path.join(os.getcwd(), "config.json")
            cm.save_config(default_path)
            print(f"Configuration saved to {default_path}")
        
        return 0
    except Exception as e:
        print(f"Error setting configuration value: {e}")
        return 1


def main(args=None):
    """
    Main entry point for the configuration CLI.
    
    Args:
        args: Command-line arguments (list of strings)
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Set up logging
    logger = setup_logging()
    
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Concept Fragmentation Configuration Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # View configuration
    view_parser = subparsers.add_parser("view", help="View configuration")
    view_parser.add_argument(
        "--format", choices=["json", "yaml", "python"], default="python",
        help="Output format"
    )
    view_parser.add_argument(
        "--output", help="Output file (if not specified, print to stdout)"
    )
    view_parser.add_argument(
        "--section", help="Only view a specific section (e.g., 'metrics.cluster_entropy')"
    )
    
    # Export configuration
    export_parser = subparsers.add_parser("export", help="Export configuration")
    export_parser.add_argument(
        "--format", choices=["json", "yaml"], default="json",
        help="Output format"
    )
    export_parser.add_argument(
        "output", help="Output file"
    )
    export_parser.add_argument(
        "--dataset", help="Dataset to export configuration for"
    )
    export_parser.add_argument(
        "--experiment", help="Experiment to export configuration for (requires --dataset)"
    )
    
    # Import configuration
    import_parser = subparsers.add_parser("import", help="Import configuration")
    import_parser.add_argument(
        "input", help="Input file (JSON or YAML)"
    )
    import_parser.add_argument(
        "--validate", action="store_true",
        help="Validate configuration before importing"
    )
    import_parser.add_argument(
        "--merge", action="store_true",
        help="Merge with current configuration instead of replacing"
    )
    import_parser.add_argument(
        "--save", action="store_true",
        help="Save the configuration to a file after importing"
    )
    
    # Validate configuration
    validate_parser = subparsers.add_parser("validate", help="Validate configuration")
    validate_parser.add_argument(
        "--input", help="Input file (JSON or YAML) to validate"
    )
    
    # Set configuration value
    set_parser = subparsers.add_parser("set", help="Set a configuration value")
    set_parser.add_argument(
        "key", help="Configuration key to set (e.g., 'random_seed' or 'metrics.cluster_entropy.default_k')"
    )
    set_parser.add_argument(
        "value", help="Value to set (will be parsed as appropriate type)"
    )
    set_parser.add_argument(
        "--save", action="store_true",
        help="Save the configuration to a file after setting the value"
    )
    
    # Parse arguments
    args = parser.parse_args(args)
    
    try:
        # Execute command
        if args.command == "view":
            return view_command(args)
        elif args.command == "export":
            return export_command(args)
        elif args.command == "import":
            return import_command(args)
        elif args.command == "validate":
            return validate_command(args)
        elif args.command == "set":
            return set_command(args)
        else:
            parser.print_help()
            return 1
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())