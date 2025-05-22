"""
Cache manager for LLM responses.

This module provides an improved caching system for LLM responses with
features like TTL, memory-only caching, and batched file writes.
"""

import os
import json
import time
import random
import hashlib
from typing import Dict, Any, Optional, Tuple, List, Union

from .responses import LLMResponse


class CacheManager:
    """Manages caching of LLM responses with improved efficiency and features."""
    
    def __init__(
        self,
        provider: str,
        model: str,
        cache_dir: str = None,
        use_cache: bool = True,
        cache_ttl: int = None,  # In seconds, None = no expiration
        memory_only: bool = False,
        save_interval: int = 10  # Save every N new items
    ):
        """
        Initialize the cache manager.
        
        Args:
            provider: LLM provider name
            model: Model name
            cache_dir: Directory to store cache files (default: ./cache/llm)
            use_cache: Whether to use caching
            cache_ttl: Time-to-live for cache entries in seconds (None = no expiration)
            memory_only: If True, don't persist cache to disk
            save_interval: Save to disk every N new cache items
        """
        self.provider = provider
        self.model = model
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl
        self.memory_only = memory_only
        self.save_interval = save_interval
        
        # Set up cache directory
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "cache", "llm")
        if self.use_cache and not self.memory_only and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize cache and metadata
        self.cache = {}  # Main cache storage
        self.cached_since_save = 0  # Counter for tracking updates
        self.last_save_time = time.time()
        
        # Load existing cache if available
        if self.use_cache and not self.memory_only:
            self._load_cache()
    
    def _get_cache_file(self) -> str:
        """Get the path to the cache file."""
        return os.path.join(self.cache_dir, f"{self.provider}_{self.model}_cache.json")
    
    def _load_cache(self) -> None:
        """Load the cache from disk with ttl cleanup."""
        cache_file = self._get_cache_file()
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                
                # Handle both old and new cache formats
                if isinstance(data, dict) and "cache" in data:
                    # New format with metadata
                    self.cache = data.get("cache", {})
                else:
                    # Old format (direct dict mapping)
                    self.cache = {}
                    for key, value in data.items():
                        # Convert old format to new format
                        if isinstance(value, dict) and not "timestamp" in value:
                            self.cache[key] = {
                                "response": value,
                                "timestamp": time.time(),  # Assume current time for old cache items
                            }
                        else:
                            self.cache[key] = value
                
                # Clean expired items if ttl is set
                if self.cache_ttl:
                    self._clean_expired_items()
                    
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load cache file: {e}")
                self.cache = {}
    
    def _clean_expired_items(self) -> None:
        """Remove expired items from cache based on ttl."""
        if not self.cache_ttl:
            return
            
        current_time = time.time()
        expired_keys = []
        
        for key, value in self.cache.items():
            # Check if item has timestamp and if it's expired
            timestamp = value.get("timestamp", 0)
            if current_time - timestamp > self.cache_ttl:
                expired_keys.append(key)
        
        # Remove expired items
        for key in expired_keys:
            del self.cache[key]
    
    def _save_cache(self, force: bool = False) -> None:
        """Save the cache to disk if conditions are met."""
        if not self.use_cache or self.memory_only:
            return
            
        # Only save if forced or enough changes accumulated
        if not force and self.cached_since_save < self.save_interval:
            return
            
        cache_file = self._get_cache_file()
        try:
            # Prepare data with metadata
            data = {
                "cache": self.cache,
                "last_update": time.time()
            }
            
            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)
                
            # Reset counter
            self.cached_since_save = 0
            self.last_save_time = time.time()
            
        except IOError as e:
            print(f"Warning: Failed to save cache file: {e}")
    
    def _generate_cache_key(self, prompt: str, **kwargs) -> str:
        """
        Generate a more robust cache key.
        
        Args:
            prompt: The prompt text
            **kwargs: Additional parameters that affect the response
            
        Returns:
            A hash-based cache key
        """
        # Filter out irrelevant kwargs
        relevant_kwargs = {
            k: v for k, v in kwargs.items() 
            if k in ['temperature', 'max_tokens', 'system_prompt']
        }
        
        # Create a stable representation of kwargs
        kwargs_items = sorted(relevant_kwargs.items())
        kwargs_str = "|".join(f"{k}={v}" for k, v in kwargs_items)
        
        # Use hash of combined prompt and kwargs for key
        key_content = f"{prompt}|{kwargs_str}|{self.model}"
        return hashlib.md5(key_content.encode()).hexdigest()
    
    def get(self, prompt: str, **kwargs) -> Optional[LLMResponse]:
        """
        Get a cached response if available.
        
        Args:
            prompt: The prompt text
            **kwargs: Additional parameters that were used in the request
            
        Returns:
            Cached LLMResponse or None if not found
        """
        if not self.use_cache:
            return None
            
        # Clean expired items occasionally
        if self.cache_ttl and random.random() < 0.1:  # ~10% chance
            self._clean_expired_items()
            
        # Generate cache key
        cache_key = self._generate_cache_key(prompt, **kwargs)
        
        # Check if key exists in cache
        if cache_key in self.cache:
            cached_item = self.cache[cache_key]
            
            # Check if item is expired
            if self.cache_ttl:
                timestamp = cached_item.get("timestamp", 0)
                if time.time() - timestamp > self.cache_ttl:
                    # Item expired, remove and return None
                    del self.cache[cache_key]
                    return None
            
            # Handle old format (direct response dict)
            if "response" not in cached_item and "text" in cached_item:
                return LLMResponse.from_dict(cached_item)
                    
            # Return cached response
            return LLMResponse.from_dict(cached_item.get("response", {}))
            
        return None
    
    def store(self, prompt: str, response: LLMResponse, **kwargs) -> None:
        """
        Store a response in the cache.
        
        Args:
            prompt: The prompt text
            response: The LLMResponse to cache
            **kwargs: Additional parameters that were used in the request
        """
        if not self.use_cache:
            return
            
        # Generate cache key
        cache_key = self._generate_cache_key(prompt, **kwargs)
        
        # Filter kwargs to only store relevant ones
        relevant_kwargs = {
            k: v for k, v in kwargs.items() 
            if k in ['temperature', 'max_tokens', 'system_prompt']
        }
        
        # Store response with metadata
        self.cache[cache_key] = {
            "response": response.to_dict(),
            "timestamp": time.time(),
            "prompt": prompt,
            "params": relevant_kwargs
        }
        
        # Update counter and check if save needed
        self.cached_since_save += 1
        
        # Auto-save based on threshold or time interval (10 min)
        if (self.cached_since_save >= self.save_interval or 
            time.time() - self.last_save_time > 600):
            self._save_cache()
    
    def clear(self, force_save: bool = True) -> None:
        """
        Clear the cache.
        
        Args:
            force_save: Whether to save the empty cache to disk
        """
        self.cache = {}
        self.cached_since_save = 0
        
        if force_save and not self.memory_only:
            self._save_cache(force=True)
    
    def get_size(self) -> int:
        """Get the number of items in the cache."""
        return len(self.cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "provider": self.provider,
            "model": self.model,
            "size": len(self.cache),
            "memory_only": self.memory_only,
            "enabled": self.use_cache
        }
        
        # Add TTL info if applicable
        if self.cache_ttl:
            stats["ttl_seconds"] = self.cache_ttl
            
            # Estimate expiring items
            current_time = time.time()
            expiring_soon = 0
            for item in self.cache.values():
                timestamp = item.get("timestamp", 0)
                if current_time - timestamp > (self.cache_ttl * 0.9):
                    expiring_soon += 1
            
            stats["expiring_soon"] = expiring_soon
        
        # Add file info if applicable
        if not self.memory_only:
            cache_file = self._get_cache_file()
            if os.path.exists(cache_file):
                stats["cache_file"] = cache_file
                stats["file_size_kb"] = os.path.getsize(cache_file) / 1024
        
        return stats
    
    def __len__(self) -> int:
        """Get the number of items in the cache."""
        return self.get_size()
    
    def close(self) -> None:
        """Ensure cache is saved before closing."""
        if self.cached_since_save > 0:
            self._save_cache(force=True)