"""
Enhanced Neural SDF model with planar priors.

Implements:
- SIREN (Sinusoidal Representation Networks) backbone
- Optional planar attention module
- Multi-head outputs (SDF, color, semantics)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict

from .encodings import PositionalEncoding, HashGridEncoding


class SineLayer(nn.Module):
    """
    SIREN layer with sinusoidal activation.
    
    Uses special initialization scheme for stable training.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to use bias
        is_first: Whether this is the first layer
        omega_0: Frequency scaling factor
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0
    ):
        super().__init__()
        
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights according to SIREN paper."""
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform in [-1/in, 1/in]
                bound = 1.0 / self.in_features
            else:
                # Hidden layers: uniform in [-sqrt(6/in)/ω₀, sqrt(6/in)/ω₀]
                bound = np.sqrt(6.0 / self.in_features) / self.omega_0
                
            self.linear.weight.uniform_(-bound, bound)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear transformation and sine activation."""
        return torch.sin(self.omega_0 * self.linear(x))


class NeuralSDF(nn.Module):
    """
    Neural Signed Distance Function network.
    
    SIREN-based architecture for learning implicit surface representation.
    
    Args:
        hidden_features: Hidden layer dimension
        hidden_layers: Number of hidden layers
        omega_0: Frequency scaling for SIREN
        encoding_type: Type of positional encoding ('none', 'positional', 'hashgrid')
        encoding_config: Configuration for encoding module
        output_color: Whether to output RGB color
        output_semantics: Whether to output semantic logits
        num_semantic_classes: Number of semantic classes
    """
    
    def __init__(
        self,
        hidden_features: int = 256,
        hidden_layers: int = 4,
        omega_0: float = 30.0,
        encoding_type: str = 'positional',
        encoding_config: Optional[Dict] = None,
        output_color: bool = True,
        output_semantics: bool = False,
        num_semantic_classes: int = 10
    ):
        super().__init__()
        
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.output_color = output_color
        self.output_semantics = output_semantics
        
        # Setup positional encoding
        if encoding_type == 'positional':
            config = encoding_config or {'num_frequencies': 10}
            self.encoding = PositionalEncoding(**config)
            input_dim = self.encoding.get_output_dim()
        elif encoding_type == 'hashgrid':
            config = encoding_config or {}
            self.encoding = HashGridEncoding(**config)
            input_dim = self.encoding.get_output_dim()
        else:
            self.encoding = None
            input_dim = 3
            
        # Build SIREN network
        layers = []
        
        # First layer
        layers.append(SineLayer(input_dim, hidden_features, is_first=True, omega_0=omega_0))
        
        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(SineLayer(hidden_features, hidden_features, omega_0=omega_0))
            
        self.net = nn.Sequential(*layers)
        
        # Output heads
        # SDF output (single value)
        self.sdf_head = nn.Linear(hidden_features, 1)
        
        # Color output (RGB)
        if output_color:
            self.color_head = nn.Sequential(
                nn.Linear(hidden_features, hidden_features // 2),
                nn.ReLU(),
                nn.Linear(hidden_features // 2, 3),
                nn.Sigmoid()
            )
            
        # Semantic output
        if output_semantics:
            self.semantic_head = nn.Linear(hidden_features, num_semantic_classes)
            
    def forward(
        self,
        coords: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            coords: (B, 3) or (B, N, 3) 3D coordinates
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing:
                - sdf: (B, 1) or (B, N, 1) signed distance values
                - color: (B, 3) or (B, N, 3) RGB colors (if enabled)
                - semantics: (B, C) or (B, N, C) semantic logits (if enabled)
                - features: (B, F) or (B, N, F) intermediate features (if requested)
        """
        original_shape = coords.shape[:-1]
        coords_flat = coords.view(-1, 3)
        
        # Apply encoding
        if self.encoding is not None:
            x = self.encoding(coords_flat)
        else:
            x = coords_flat
            
        # Forward through SIREN backbone
        features = self.net(x)
        
        # Compute outputs
        outputs = {}
        
        # SDF
        sdf = self.sdf_head(features)
        outputs['sdf'] = sdf.view(*original_shape, 1)
        
        # Color
        if self.output_color:
            color = self.color_head(features)
            outputs['color'] = color.view(*original_shape, 3)
            
        # Semantics
        if self.output_semantics:
            semantics = self.semantic_head(features)
            outputs['semantics'] = semantics.view(*original_shape, -1)
            
        # Features
        if return_features:
            outputs['features'] = features.view(*original_shape, self.hidden_features)
            
        return outputs
        
    def gradient(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of SDF w.r.t. coordinates.
        
        Args:
            coords: (B, 3) 3D coordinates
            
        Returns:
            gradient: (B, 3) SDF gradient (surface normal direction)
        """
        coords = coords.requires_grad_(True)
        outputs = self.forward(coords)
        sdf = outputs['sdf']
        
        gradient = torch.autograd.grad(
            sdf,
            coords,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True,
            retain_graph=True
        )[0]
        
        return gradient
        
    def sdf_and_gradient(
        self,
        coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute SDF value and gradient simultaneously.
        
        Args:
            coords: (B, 3) 3D coordinates
            
        Returns:
            sdf: (B, 1) SDF values
            gradient: (B, 3) SDF gradients
        """
        coords = coords.requires_grad_(True)
        outputs = self.forward(coords)
        sdf = outputs['sdf']
        
        gradient = torch.autograd.grad(
            sdf,
            coords,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True,
            retain_graph=True
        )[0]
        
        return sdf, gradient


class NeuralSDFWithPlanar(NeuralSDF):
    """
    Neural SDF with planar attention module.
    
    Incorporates detected plane features to improve reconstruction
    of flat surfaces in indoor environments.
    
    Args:
        planar_attention_config: Configuration for planar attention
        **kwargs: Arguments passed to NeuralSDF
    """
    
    def __init__(
        self,
        planar_attention_config: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Import here to avoid circular dependency
        from .planar_attention import PlanarAttention
        
        config = planar_attention_config or {}
        self.planar_attention = PlanarAttention(
            embed_dim=self.hidden_features,
            **config
        )
        
        # Modify SDF head to accept planar-enhanced features
        self.sdf_head = nn.Sequential(
            nn.Linear(self.hidden_features * 2, self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features, 1)
        )
        
    def forward(
        self,
        coords: torch.Tensor,
        plane_features: Optional[torch.Tensor] = None,
        plane_mask: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with planar attention.
        
        Args:
            coords: (B, 3) or (B, N, 3) 3D coordinates
            plane_features: (B, P, F) features of P detected planes
            plane_mask: (B, P) valid plane mask
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary with sdf, color, semantics, features
        """
        original_shape = coords.shape[:-1]
        coords_flat = coords.view(-1, 3)
        
        # Apply encoding
        if self.encoding is not None:
            x = self.encoding(coords_flat)
        else:
            x = coords_flat
            
        # Forward through SIREN backbone
        features = self.net(x)
        features = features.view(*original_shape, self.hidden_features)
        
        # Apply planar attention if plane features are provided
        if plane_features is not None:
            planar_features = self.planar_attention(
                features, plane_features, plane_mask
            )
            # Concatenate original and planar-enhanced features
            combined = torch.cat([features, planar_features], dim=-1)
        else:
            # No planes: duplicate features for consistent dimension
            combined = torch.cat([features, features], dim=-1)
            
        # Compute outputs
        outputs = {}
        
        # SDF from combined features
        combined_flat = combined.view(-1, self.hidden_features * 2)
        sdf = self.sdf_head(combined_flat)
        outputs['sdf'] = sdf.view(*original_shape, 1)
        
        # Color (from original features)
        if self.output_color:
            features_flat = features.view(-1, self.hidden_features)
            color = self.color_head(features_flat)
            outputs['color'] = color.view(*original_shape, 3)
            
        # Semantics
        if self.output_semantics:
            features_flat = features.view(-1, self.hidden_features)
            semantics = self.semantic_head(features_flat)
            outputs['semantics'] = semantics.view(*original_shape, -1)
            
        if return_features:
            outputs['features'] = features
            outputs['planar_features'] = planar_features if plane_features is not None else None
            
        return outputs
