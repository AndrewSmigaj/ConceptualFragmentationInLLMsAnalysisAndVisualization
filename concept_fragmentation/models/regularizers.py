"""
Regularization implementations for concept fragmentation analysis.
This module provides regularization techniques to mitigate concept fragmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

from ..config import REGULARIZATION, RANDOM_SEED


class CohesionRegularizer:
    """
    Cohesion regularization implementation to reduce concept fragmentation.
    
    This regularizer implements a contrastive loss term that encourages
    same-class samples to have similar representations.
    
    Features:
    - Contrastive loss term for same-class pairs
    - Minibatch threshold parameter for selecting pairs
    - Weighting parameter for regularization strength
    """
    
    def __init__(
        self,
        weight: float = REGULARIZATION["cohesion"]["weight"],
        temperature: float = REGULARIZATION["cohesion"]["temperature"],
        threshold: float = REGULARIZATION["cohesion"]["similarity_threshold"],
        minibatch_size: int = REGULARIZATION["cohesion"]["minibatch_size"],
        seed: int = RANDOM_SEED
    ):
        """
        Initialize the cohesion regularizer.
        
        Args:
            weight: Weight of the regularization term in the overall loss
            temperature: Temperature parameter for the contrastive loss
            threshold: Similarity threshold for considering samples as positive pairs
            minibatch_size: Number of samples to use in each minibatch
            seed: Random seed for reproducibility
        """
        self.weight = weight
        self.temperature = temperature
        self.threshold = threshold
        self.minibatch_size = minibatch_size
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def compute_loss(
        self, 
        activations: torch.Tensor, 
        labels: torch.Tensor,
        layer_name: str = 'layer3'
    ) -> torch.Tensor:
        """
        Compute the cohesion regularization loss.
        
        Args:
            activations: Tensor of shape (batch_size, feature_dim) containing 
                         the activations from a specific layer
            labels: Tensor of shape (batch_size,) containing the class labels
            layer_name: Name of the layer to apply regularization to (for logging)
            
        Returns:
            Cohesion regularization loss
        """
        batch_size = activations.size(0)
        
        # If batch is too small, return zero loss
        if batch_size < 2:
            return torch.tensor(0.0, device=activations.device)
        
        # Normalize activations
        normalized_activations = F.normalize(activations, p=2, dim=1)
        
        # Compute pairwise cosine similarities
        similarities = torch.matmul(normalized_activations, normalized_activations.t())
        
        # Create mask for same-class pairs
        mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
        
        # Remove self-similarities (diagonal elements)
        mask.fill_diagonal_(False)
        
        # If no same-class pairs, return zero loss
        if not torch.any(mask):
            return torch.tensor(0.0, device=activations.device)
        
        # Scale similarities by temperature
        scaled_similarities = similarities / self.temperature
        
        # Create positive and negative masks
        positive_mask = mask & (similarities > self.threshold)
        negative_mask = ~mask
        
        # If no positive pairs after threshold, return zero loss
        if not torch.any(positive_mask):
            return torch.tensor(0.0, device=activations.device)
        
        # Compute contrastive loss
        # For each positive pair, compute the InfoNCE loss term
        loss = 0.0
        num_positive_pairs = positive_mask.sum().item()
        
        # Use minibatch selection if the number of positive pairs is too large
        if num_positive_pairs > self.minibatch_size:
            # Flatten the positive mask and get indices of True elements
            positive_indices = torch.nonzero(positive_mask.view(-1), as_tuple=True)[0]
            # Randomly select minibatch_size indices
            selected_indices = positive_indices[torch.randperm(len(positive_indices))[:self.minibatch_size]]
            # Convert flat indices back to 2D indices
            rows = selected_indices // batch_size
            cols = selected_indices % batch_size
            # Create a new positive mask with only the selected pairs
            minibatch_mask = torch.zeros_like(positive_mask)
            minibatch_mask[rows, cols] = True
            positive_mask = minibatch_mask
            num_positive_pairs = self.minibatch_size
        
        # Extract positive pairs
        pos_i, pos_j = torch.where(positive_mask)
        
        # Compute loss for each positive pair
        for i, j in zip(pos_i, pos_j):
            # Positive similarity
            pos_sim = scaled_similarities[i, j]
            
            # Negative similarities (all samples from different classes)
            neg_mask = negative_mask[i]
            
            if not torch.any(neg_mask):
                continue  # Skip if no negative samples for this anchor
                
            neg_sims = scaled_similarities[i][neg_mask]
            
            # Compute the log sum exp term in the denominator
            neg_term = torch.logsumexp(neg_sims, dim=0)
            
            # Compute the InfoNCE loss for this positive pair
            pair_loss = -pos_sim + neg_term
            loss += pair_loss
        
        # Average the loss over all positive pairs
        if num_positive_pairs > 0:
            loss = loss / num_positive_pairs
        
        # Apply the weight
        weighted_loss = self.weight * loss
        
        return weighted_loss
    
    def __call__(
        self, 
        activations_dict: Dict[str, torch.Tensor], 
        labels: torch.Tensor,
        layer_name: str = 'layer3'
    ) -> torch.Tensor:
        """
        Compute the cohesion regularization loss for a given layer.
        
        Args:
            activations_dict: Dictionary mapping layer names to activation tensors
            labels: Tensor of shape (batch_size,) containing the class labels
            layer_name: Name of the layer to apply regularization to
            
        Returns:
            Cohesion regularization loss
        """
        if layer_name not in activations_dict:
            raise ValueError(f"Layer {layer_name} not found in activations dictionary")
        
        activations = activations_dict[layer_name]
        return self.compute_loss(activations, labels, layer_name)
    
    def multi_layer_loss(
        self,
        activations_dict: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        layer_names: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute cohesion regularization loss for multiple layers.
        
        Args:
            activations_dict: Dictionary mapping layer names to activation tensors
            labels: Tensor of shape (batch_size,) containing the class labels
            layer_names: List of layer names to apply regularization to
                        If None, applies to all layers in activations_dict
            
        Returns:
            Dictionary mapping layer names to regularization losses
        """
        if layer_names is None:
            layer_names = list(activations_dict.keys())
        
        losses = {}
        for layer_name in layer_names:
            if layer_name in activations_dict:
                losses[layer_name] = self.compute_loss(
                    activations_dict[layer_name], 
                    labels, 
                    layer_name
                )
            else:
                losses[layer_name] = torch.tensor(0.0, device=labels.device)
        
        return losses
    
    def get_total_loss(
        self,
        activations_dict: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        layer_names: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Get the total cohesion regularization loss across multiple layers.
        
        Args:
            activations_dict: Dictionary mapping layer names to activation tensors
            labels: Tensor of shape (batch_size,) containing the class labels
            layer_names: List of layer names to apply regularization to
                        If None, applies to all layers in activations_dict
            
        Returns:
            Total regularization loss
        """
        losses = self.multi_layer_loss(activations_dict, labels, layer_names)
        return sum(losses.values())
