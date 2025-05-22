"""
Prompt optimization utilities for reducing token usage.

This module provides utilities for optimizing prompts to reduce token usage
while maintaining the quality of the responses.
"""

import re
from typing import Dict, Any, List, Optional, Union


class PromptOptimizer:
    """
    Utility class for optimizing prompts to reduce token usage.
    
    Optimization strategies:
    - Removing unnecessary whitespace
    - Compressing verbose instructions
    - Trimming examples to essential parts
    - Using more concise language
    """
    
    # Common patterns that can be made more concise
    VERBOSE_PATTERNS = {
        # Instructions
        r"You are an AI (expert|assistant) (that|who) (is )?(analyzing|helping with)": "As an AI analyzing",
        r"I want you to (provide|give me|generate)": "Please provide",
        r"Please provide a (detailed|comprehensive) (analysis|explanation)": "Please analyze",
        r"For each (item|element|point|case), (please )?(provide|include|add)": "For each item, add",
        
        # Redundancies
        r"Please note that ": "",
        r"Keep in mind that ": "",
        r"It('s| is) important to (remember|note|consider) that ": "",
        r" in order to ": " to ",
        r"for the purpose of ": "for ",
        r"due to the fact that ": "because ",
        r"in the event that ": "if ",
        
        # Verbosity
        r"very ": "",
        r"extremely ": "",
        r"absolutely ": "",
        r"basically ": "",
        r"actually ": "",
    }
    
    # Phrasings that can be shortened 
    CONCISE_ALTERNATIVES = {
        "neural network activation patterns": "activations",
        "a human-readable narrative": "a narrative",
        "a human-readable label": "a label",
        "a concise, meaningful label": "a label",
        "in the neural network": "",
        "based on the information provided": "",
        "for the purpose of analysis": "",
        "in this context": "",
        "with respect to": "regarding",
    }
    
    @classmethod
    def optimize_prompt(cls, prompt: str, optimization_level: int = 1) -> str:
        """
        Optimize a prompt to reduce token usage.
        
        Args:
            prompt: The prompt to optimize
            optimization_level: Level of optimization (1-3, higher = more aggressive)
            
        Returns:
            Optimized prompt
        """
        if optimization_level <= 0:
            return prompt
        
        # Start with basic optimizations
        optimized = cls._remove_redundant_whitespace(prompt)
        
        # Apply pattern replacements based on optimization level
        if optimization_level >= 1:
            # Level 1: Replace obvious verbose patterns
            optimized = cls._replace_verbose_patterns(optimized)
            optimized = cls._replace_concise_alternatives(optimized)
        
        if optimization_level >= 2:
            # Level 2: More aggressive trimming
            optimized = cls._compress_instructions(optimized)
            optimized = cls._trim_examples(optimized)
        
        if optimization_level >= 3:
            # Level 3: Maximum compression
            optimized = cls._remove_politeness(optimized)
            optimized = cls._compact_structure(optimized)
        
        return optimized
    
    @staticmethod
    def _remove_redundant_whitespace(text: str) -> str:
        """Remove redundant whitespace from text."""
        # Replace multiple newlines with a single newline
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r" {2,}", " ", text)
        
        # Remove whitespace at the beginning of lines
        text = re.sub(r"(?m)^ +", "", text)
        
        # Remove trailing whitespace
        text = re.sub(r"(?m) +$", "", text)
        
        return text.strip()
    
    @classmethod
    def _replace_verbose_patterns(cls, text: str) -> str:
        """Replace verbose patterns with more concise alternatives."""
        for pattern, replacement in cls.VERBOSE_PATTERNS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text
    
    @classmethod
    def _replace_concise_alternatives(cls, text: str) -> str:
        """Replace phrases with more concise alternatives."""
        for phrase, replacement in cls.CONCISE_ALTERNATIVES.items():
            text = text.replace(phrase, replacement)
        return text
    
    @staticmethod
    def _compress_instructions(text: str) -> str:
        """Compress verbose instructions into more concise form."""
        # Identify instruction blocks (usually at the beginning)
        instruction_pattern = r"(?s)(^.*?)((?:\n\n|\n\w{2,}:).*$)"
        match = re.match(instruction_pattern, text)
        
        if match:
            instructions, remainder = match.groups()
            
            # Compress instructions
            compressed = instructions
            # Remove self-references
            compressed = re.sub(r"I want you to|You should|Your task is to", "", compressed)
            # Remove meta-instructions about response format unless they're crucial
            compressed = re.sub(r"Your response should be|Format your response as|Make sure to include", "", compressed)
            
            return compressed.strip() + remainder
        
        return text
    
    @staticmethod
    def _trim_examples(text: str) -> str:
        """Trim examples to their essential parts."""
        # Look for example sections
        example_pattern = r"(?s)(Example[s]?:.*?)((?:\n\n[A-Z]|\Z))"
        match = re.search(example_pattern, text)
        
        if match:
            example_section = match.group(1)
            
            # Keep only first example if multiple exist
            examples = re.split(r"\n\s*Example \d+:", example_section)
            if len(examples) > 2:  # First item is the "Examples:" header
                trimmed_examples = examples[0] + examples[1]
                text = text.replace(example_section, trimmed_examples)
        
        return text
    
    @staticmethod
    def _remove_politeness(text: str) -> str:
        """Remove unnecessary politeness phrases."""
        # Remove phrases like "please", "thank you", etc.
        text = re.sub(r"(?i)\b(please|kindly|thanks|thank you)\b", "", text)
        return text
    
    @staticmethod
    def _compact_structure(text: str) -> str:
        """Make the structure more compact."""
        # Convert bullet points to more compact form
        text = re.sub(r"\n\s*-\s+", "\n- ", text)
        
        # Convert numbered lists to more compact form 
        text = re.sub(r"\n\s*(\d+)\.\s+", "\n\\1. ", text)
        
        return text


def optimize_cluster_label_prompt(
    prompt: str,
    optimization_level: int = 1
) -> str:
    """
    Optimize a cluster labeling prompt.
    
    Args:
        prompt: The original prompt
        optimization_level: Level of optimization (1-3)
        
    Returns:
        Optimized prompt
    """
    # Apply general optimizations first
    optimized = PromptOptimizer.optimize_prompt(prompt, optimization_level)
    
    # For higher optimization levels, apply more specific optimizations
    if optimization_level >= 2:
        # Focus on the essential instructions for labeling
        optimized = re.sub(
            r"(?i)You are an AI expert analyzing neural network activations\.\s+",
            "",
            optimized
        )
        
        # Simplify the labeling instructions
        optimized = re.sub(
            r"(?i)provide a concise, meaningful label.*?concept this cluster might represent\.",
            "provide a brief label (1-5 words) for this cluster.",
            optimized
        )
    
    return optimized


def optimize_path_narrative_prompt(
    prompt: str,
    optimization_level: int = 1
) -> str:
    """
    Optimize a path narrative prompt.
    
    Args:
        prompt: The original prompt
        optimization_level: Level of optimization (1-3)
        
    Returns:
        Optimized prompt
    """
    # Apply general optimizations first
    optimized = PromptOptimizer.optimize_prompt(prompt, optimization_level)
    
    # For higher optimization levels, apply more specific optimizations
    if optimization_level >= 2:
        # Remove the expert preamble
        optimized = re.sub(
            r"(?i)You are an AI expert analyzing neural network activation patterns\.\s+",
            "",
            optimized
        )
        
        # Simplify the narrative instructions
        optimized = re.sub(
            r"(?i)Generate a clear, insightful narrative.*?decision process represented by this path\.",
            "Explain this activation path briefly:",
            optimized
        )
        
        # Simplify the conclusion
        optimized = re.sub(
            r"(?i)Your explanation should be clear and insightful without being overly technical\.",
            "",
            optimized
        )
    
    return optimized


def estimate_token_savings(original: str, optimized: str) -> Dict[str, Any]:
    """
    Estimate token savings from optimization.
    
    Args:
        original: Original prompt
        optimized: Optimized prompt
        
    Returns:
        Dictionary with token saving estimates
    """
    # Very rough token estimation (average 4 chars per token)
    original_tokens = len(original) / 4
    optimized_tokens = len(optimized) / 4
    token_savings = original_tokens - optimized_tokens
    
    return {
        "original_length": len(original),
        "optimized_length": len(optimized),
        "original_tokens_estimate": int(original_tokens),
        "optimized_tokens_estimate": int(optimized_tokens),
        "tokens_saved_estimate": int(token_savings),
        "percent_reduction": round((token_savings / original_tokens) * 100, 1)
    }