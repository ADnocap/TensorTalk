"""
k-NN Matcher Module for TensorTalk

This module implements k-Nearest Neighbors matching for voice conversion using cosine similarity.
The matcher finds the k most similar feature vectors from the target speaker and uses them
to replace source speaker features while preserving phonetic content.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


class KNNMatcher:
    """
    k-Nearest Neighbors matcher for voice conversion.
    
    Uses cosine similarity to find the k most similar features from the target speaker
    for each frame of the source speech. This approach preserves phonetic content while
    transferring speaker characteristics.
    
    Attributes:
        k (int): Number of nearest neighbors to consider
        device (str): Device to run computations on
    """
    
    def __init__(self, k: int = 4, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the k-NN matcher.
        
        Args:
            k: Number of nearest neighbors to use for matching
            device: Device to use for computations
        """
        self.k = k
        self.device = device
    
    def cosine_similarity(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between source and target features.
        
        Args:
            source: Source features of shape [T_source, feature_dim]
            target: Target features of shape [T_target, feature_dim]
            
        Returns:
            Similarity matrix of shape [T_source, T_target]
        """
        # Normalize features
        source_norm = F.normalize(source, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.mm(source_norm, target_norm.t())
        
        return similarity
    
    def match_features(
        self, 
        source_features: torch.Tensor, 
        target_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Match source features to target features using k-NN.
        
        For each source feature vector, finds the k most similar vectors from the
        target features and returns their average.
        
        Args:
            source_features: Source speaker features [T_source, feature_dim]
            target_features: Target speaker features [T_target, feature_dim]
            
        Returns:
            Tuple of:
                - matched_features: Matched target features [T_source, feature_dim]
                - distances: Cosine distances to nearest neighbors [T_source, k]
        """
        # Ensure features are on the correct device
        source_features = source_features.to(self.device)
        target_features = target_features.to(self.device)
        
        # Compute cosine similarity
        similarity = self.cosine_similarity(source_features, target_features)
        
        # Find k nearest neighbors
        distances, indices = torch.topk(similarity, k=self.k, dim=1, largest=True)
        
        # Gather the k nearest neighbor features
        matched_features = target_features[indices]  # Shape: [T_source, k, feature_dim]
        
        # Average the k nearest neighbors
        matched_features = matched_features.mean(dim=1)  # Shape: [T_source, feature_dim]
        
        return matched_features, distances
    
    def interpolate_features(
        self, 
        source_features: torch.Tensor, 
        matched_features: torch.Tensor, 
        alpha: float = 1.0
    ) -> torch.Tensor:
        """
        Linearly interpolate between source and matched features.
        
        Args:
            source_features: Original source features [T, feature_dim]
            matched_features: Matched target features [T, feature_dim]
            alpha: Interpolation factor (0 = source only, 1 = target only)
            
        Returns:
            Interpolated features [T, feature_dim]
        """
        alpha = torch.clamp(torch.tensor(alpha), 0.0, 1.0).to(self.device)
        
        converted_features = alpha * matched_features + (1 - alpha) * source_features
        
        return converted_features
    
    def convert_voice(
        self, 
        source_features: torch.Tensor, 
        target_features: torch.Tensor, 
        alpha: float = 1.0
    ) -> torch.Tensor:
        """
        Complete voice conversion pipeline.
        
        Args:
            source_features: Source speaker features [T_source, feature_dim]
            target_features: Target speaker features [T_target, feature_dim]
            alpha: Interpolation factor for blending
            
        Returns:
            Converted features [T_source, feature_dim]
        """
        # Match features using k-NN
        matched_features, _ = self.match_features(source_features, target_features)
        
        # Interpolate between source and matched features
        converted_features = self.interpolate_features(
            source_features, matched_features, alpha
        )
        
        return converted_features
