"""
GPT-2 Analysis Results Persistence Manager.

This module provides comprehensive data persistence functionality for GPT-2 analysis results,
including saving, loading, and caching of analysis data, visualizations, and user sessions.
"""

import os
import json
import pickle
import time
import hashlib
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np

# Import cache manager for consistency
from concept_fragmentation.llm.cache_manager import CacheManager


class GPT2AnalysisPersistence:
    """Manages persistence of GPT-2 analysis results with versioning and caching."""
    
    def __init__(
        self,
        base_dir: str = None,
        enable_cache: bool = True,
        cache_ttl: int = 3600,  # 1 hour
        max_versions: int = 10
    ):
        """
        Initialize the GPT-2 analysis persistence manager.
        
        Args:
            base_dir: Base directory for storing analysis results
            enable_cache: Whether to enable caching
            cache_ttl: Time-to-live for cached results in seconds
            max_versions: Maximum number of versions to keep for each analysis
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd() / "data" / "gpt2_analysis"
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        self.max_versions = max_versions
        
        # Create directory structure
        self._create_directories()
        
        # Initialize cache
        if self.enable_cache:
            self.cache = CacheManager(
                provider="gpt2_analysis",
                model="persistence",
                cache_dir=str(self.base_dir / "cache"),
                use_cache=True,
                cache_ttl=cache_ttl,
                memory_only=False,
                save_interval=5
            )
        else:
            self.cache = None
        
        # Track active sessions
        self.active_sessions = {}
    
    def _create_directories(self):
        """Create necessary directory structure."""
        directories = [
            self.base_dir,
            self.base_dir / "analysis_results",
            self.base_dir / "visualizations",
            self.base_dir / "sessions",
            self.base_dir / "cache",
            self.base_dir / "exports",
            self.base_dir / "metadata"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _generate_analysis_id(self, model_name: str, input_text: str, timestamp: Optional[str] = None) -> str:
        """Generate a unique ID for an analysis."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Create hash from model name and input text
        content = f"{model_name}_{input_text}_{timestamp}"
        hash_obj = hashlib.md5(content.encode())
        
        return f"{model_name}_{hash_obj.hexdigest()[:8]}_{timestamp.replace(':', '-').replace('.', '-')}"
    
    def save_analysis_results(
        self,
        analysis_data: Dict[str, Any],
        model_name: str,
        input_text: str,
        analysis_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save complete GPT-2 analysis results.
        
        Args:
            analysis_data: Complete analysis results including token paths, attention data, etc.
            model_name: Name of the GPT-2 model analyzed
            input_text: Original input text
            analysis_id: Optional custom analysis ID
            metadata: Optional additional metadata
            
        Returns:
            Analysis ID for the saved results
        """
        # Generate analysis ID if not provided
        if analysis_id is None:
            analysis_id = self._generate_analysis_id(model_name, input_text)
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "model_name": model_name,
            "input_text": input_text,
            "analysis_id": analysis_id,
            "timestamp": datetime.now().isoformat(),
            "version": 1,
            "data_format": "gpt2_analysis_v1"
        })
        
        # Check for existing versions
        existing_versions = self._get_existing_versions(analysis_id)
        if existing_versions:
            metadata["version"] = max(existing_versions) + 1
        
        # Prepare complete data structure
        complete_data = {
            "metadata": metadata,
            "analysis_data": analysis_data,
            "model_info": {
                "model_name": model_name,
                "input_text": input_text,
                "input_length": len(input_text),
                "num_tokens": len(analysis_data.get("token_metadata", {}).get("tokens", [])),
                "num_layers": len(analysis_data.get("layers", []))
            }
        }
        
        # Save to file
        filename = f"{analysis_id}_v{metadata['version']}.json"
        filepath = self.base_dir / "analysis_results" / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(complete_data, f, indent=2, ensure_ascii=False, default=self._json_serializer)
            
            # Save metadata separately for quick lookup
            self._save_metadata(analysis_id, metadata)
            
            # Update cache
            if self.cache:
                cache_key = f"analysis_{analysis_id}"
                self.cache.store(cache_key, complete_data)
            
            # Cleanup old versions
            self._cleanup_old_versions(analysis_id)
            
            print(f"Saved GPT-2 analysis results: {analysis_id} (version {metadata['version']})")
            return analysis_id
            
        except Exception as e:
            print(f"Error saving analysis results: {e}")
            raise
    
    def load_analysis_results(self, analysis_id: str, version: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Load GPT-2 analysis results.
        
        Args:
            analysis_id: ID of the analysis to load
            version: Specific version to load (latest if None)
            
        Returns:
            Complete analysis data or None if not found
        """
        # Check cache first
        if self.cache:
            cache_key = f"analysis_{analysis_id}"
            if version is None:  # Only use cache for latest version
                cached_data = self.cache.get(cache_key)
                if cached_data:
                    return cached_data
        
        # Determine version to load
        if version is None:
            existing_versions = self._get_existing_versions(analysis_id)
            if not existing_versions:
                return None
            version = max(existing_versions)
        
        # Load from file
        filename = f"{analysis_id}_v{version}.json"
        filepath = self.base_dir / "analysis_results" / filename
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Update cache if loading latest version
            if self.cache and version == max(self._get_existing_versions(analysis_id)):
                cache_key = f"analysis_{analysis_id}"
                self.cache.store(cache_key, data)
            
            return data
            
        except Exception as e:
            print(f"Error loading analysis results: {e}")
            return None
    
    def save_visualization_state(
        self,
        analysis_id: str,
        visualization_config: Dict[str, Any],
        visualization_type: str,
        state_name: str = "default"
    ) -> str:
        """
        Save visualization state for later restoration.
        
        Args:
            analysis_id: ID of the associated analysis
            visualization_config: Configuration of the visualization
            visualization_type: Type of visualization (e.g., "token_sankey", "attention_flow")
            state_name: Name for this state
            
        Returns:
            State ID for the saved visualization state
        """
        state_id = f"{analysis_id}_{visualization_type}_{state_name}"
        
        state_data = {
            "state_id": state_id,
            "analysis_id": analysis_id,
            "visualization_type": visualization_type,
            "state_name": state_name,
            "config": visualization_config,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to file
        filepath = self.base_dir / "visualizations" / f"{state_id}.json"
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False, default=self._json_serializer)
            
            return state_id
            
        except Exception as e:
            print(f"Error saving visualization state: {e}")
            raise
    
    def load_visualization_state(self, state_id: str) -> Optional[Dict[str, Any]]:
        """Load visualization state."""
        filepath = self.base_dir / "visualizations" / f"{state_id}.json"
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading visualization state: {e}")
            return None
    
    def create_session(self, session_name: str, analysis_ids: List[str] = None) -> str:
        """
        Create a new analysis session.
        
        Args:
            session_name: Name for the session
            analysis_ids: List of analysis IDs to include in the session
            
        Returns:
            Session ID
        """
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(session_name.encode()).hexdigest()[:8]}"
        
        session_data = {
            "session_id": session_id,
            "session_name": session_name,
            "created_at": datetime.now().isoformat(),
            "analysis_ids": analysis_ids or [],
            "visualization_states": [],
            "notes": "",
            "tags": []
        }
        
        # Save session
        filepath = self.base_dir / "sessions" / f"{session_id}.json"
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            # Track active session
            self.active_sessions[session_id] = session_data
            
            return session_id
            
        except Exception as e:
            print(f"Error creating session: {e}")
            raise
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load an analysis session."""
        filepath = self.base_dir / "sessions" / f"{session_id}.json"
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            # Track as active session
            self.active_sessions[session_id] = session_data
            
            return session_data
            
        except Exception as e:
            print(f"Error loading session: {e}")
            return None
    
    def export_analysis(
        self,
        analysis_id: str,
        export_format: str = "json",
        include_visualizations: bool = True,
        output_path: Optional[str] = None
    ) -> str:
        """
        Export analysis results in various formats.
        
        Args:
            analysis_id: ID of the analysis to export
            export_format: Format for export ("json", "csv", "pickle")
            include_visualizations: Whether to include visualization states
            output_path: Optional custom output path
            
        Returns:
            Path to the exported file
        """
        # Load analysis data
        analysis_data = self.load_analysis_results(analysis_id)
        if not analysis_data:
            raise ValueError(f"Analysis {analysis_id} not found")
        
        # Prepare export data
        export_data = {
            "analysis_results": analysis_data,
            "export_metadata": {
                "analysis_id": analysis_id,
                "export_format": export_format,
                "export_timestamp": datetime.now().isoformat(),
                "include_visualizations": include_visualizations
            }
        }
        
        # Add visualization states if requested
        if include_visualizations:
            viz_states = self._get_visualization_states_for_analysis(analysis_id)
            export_data["visualization_states"] = viz_states
        
        # Determine output path
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{analysis_id}_export_{timestamp}.{export_format}"
            output_path = self.base_dir / "exports" / filename
        else:
            output_path = Path(output_path)
        
        # Export in specified format
        try:
            if export_format == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False, default=self._json_serializer)
            
            elif export_format == "pickle":
                with open(output_path, 'wb') as f:
                    pickle.dump(export_data, f)
            
            elif export_format == "csv":
                # For CSV, we'll export key metrics and token data
                self._export_to_csv(export_data, output_path)
            
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
            
            print(f"Exported analysis {analysis_id} to {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"Error exporting analysis: {e}")
            raise
    
    def list_analyses(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available analyses, optionally filtered by model name."""
        analyses = []
        
        # Get all metadata files
        metadata_dir = self.base_dir / "metadata"
        for metadata_file in metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Filter by model name if specified
                if model_name is None or metadata.get("model_name") == model_name:
                    analyses.append(metadata)
                    
            except Exception as e:
                print(f"Error reading metadata file {metadata_file}: {e}")
        
        # Sort by timestamp (newest first)
        analyses.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return analyses
    
    def cleanup_cache(self, max_age_hours: int = 24):
        """Clean up cached data older than specified age."""
        if self.cache:
            self.cache.clear()
        
        # Clean up old temporary files
        cache_dir = self.base_dir / "cache"
        if cache_dir.exists():
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            for file_path in cache_dir.glob("*"):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        try:
                            file_path.unlink()
                        except Exception as e:
                            print(f"Error cleaning up cache file {file_path}: {e}")
    
    def _get_existing_versions(self, analysis_id: str) -> List[int]:
        """Get list of existing versions for an analysis."""
        versions = []
        pattern = f"{analysis_id}_v*.json"
        
        for filepath in (self.base_dir / "analysis_results").glob(pattern):
            try:
                # Extract version number from filename
                version_str = filepath.stem.split("_v")[-1]
                version = int(version_str)
                versions.append(version)
            except (ValueError, IndexError):
                continue
        
        return sorted(versions)
    
    def _cleanup_old_versions(self, analysis_id: str):
        """Remove old versions beyond max_versions limit."""
        versions = self._get_existing_versions(analysis_id)
        
        if len(versions) > self.max_versions:
            # Remove oldest versions
            versions_to_remove = sorted(versions)[:-self.max_versions]
            
            for version in versions_to_remove:
                filename = f"{analysis_id}_v{version}.json"
                filepath = self.base_dir / "analysis_results" / filename
                
                try:
                    filepath.unlink()
                    print(f"Removed old version: {filename}")
                except Exception as e:
                    print(f"Error removing old version {filename}: {e}")
    
    def _save_metadata(self, analysis_id: str, metadata: Dict[str, Any]):
        """Save metadata for quick lookup."""
        metadata_path = self.base_dir / "metadata" / f"{analysis_id}.json"
        
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving metadata: {e}")
    
    def _get_visualization_states_for_analysis(self, analysis_id: str) -> List[Dict[str, Any]]:
        """Get all visualization states associated with an analysis."""
        states = []
        
        for state_file in (self.base_dir / "visualizations").glob(f"{analysis_id}_*.json"):
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
                    states.append(state_data)
            except Exception as e:
                print(f"Error loading visualization state {state_file}: {e}")
        
        return states
    
    def _export_to_csv(self, export_data: Dict[str, Any], output_path: Path):
        """Export analysis data to CSV format."""
        import pandas as pd
        
        analysis_data = export_data["analysis_results"]["analysis_data"]
        
        # Create multiple CSV files for different data types
        base_path = output_path.with_suffix('')
        
        # Export token metadata
        if "token_metadata" in analysis_data:
            token_metadata = analysis_data["token_metadata"]
            if "tokens" in token_metadata:
                token_df = pd.DataFrame({
                    "token": token_metadata["tokens"],
                    "position": token_metadata.get("positions", list(range(len(token_metadata["tokens"]))))
                })
                token_df.to_csv(f"{base_path}_tokens.csv", index=False)
        
        # Export token paths
        if "token_paths" in analysis_data:
            token_paths = analysis_data["token_paths"]
            path_records = []
            
            for token_id, path_data in token_paths.items():
                record = {
                    "token_id": token_id,
                    "position": path_data.get("position", ""),
                    "path_length": path_data.get("path_length", ""),
                    "cluster_changes": path_data.get("cluster_changes", ""),
                    "mobility_score": path_data.get("mobility_score", "")
                }
                path_records.append(record)
            
            if path_records:
                paths_df = pd.DataFrame(path_records)
                paths_df.to_csv(f"{base_path}_paths.csv", index=False)
        
        # Export layer information
        if "layers" in analysis_data:
            layers_df = pd.DataFrame({"layer": analysis_data["layers"]})
            layers_df.to_csv(f"{base_path}_layers.csv", index=False)
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy arrays and other objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# Convenience functions for common operations
def save_gpt2_analysis(
    analysis_data: Dict[str, Any],
    model_name: str,
    input_text: str,
    persistence_manager: Optional[GPT2AnalysisPersistence] = None
) -> str:
    """
    Convenience function to save GPT-2 analysis results.
    
    Args:
        analysis_data: Analysis results data
        model_name: GPT-2 model name
        input_text: Input text
        persistence_manager: Optional persistence manager instance
        
    Returns:
        Analysis ID
    """
    if persistence_manager is None:
        persistence_manager = GPT2AnalysisPersistence()
    
    return persistence_manager.save_analysis_results(
        analysis_data=analysis_data,
        model_name=model_name,
        input_text=input_text
    )


def load_gpt2_analysis(
    analysis_id: str,
    persistence_manager: Optional[GPT2AnalysisPersistence] = None
) -> Optional[Dict[str, Any]]:
    """
    Convenience function to load GPT-2 analysis results.
    
    Args:
        analysis_id: Analysis ID to load
        persistence_manager: Optional persistence manager instance
        
    Returns:
        Analysis data or None if not found
    """
    if persistence_manager is None:
        persistence_manager = GPT2AnalysisPersistence()
    
    return persistence_manager.load_analysis_results(analysis_id)