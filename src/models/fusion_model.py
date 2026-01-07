"""
Multi-Modal Fusion Model for Stroke Classification

This module implements a trimodal fusion network that combines:
1. Vision Transformer (ViT) for brain MRI/CT scan analysis
2. Biomarker MLP for clinical risk assessment
3. EEG CNN for brain activity monitoring

The fusion strategy uses late fusion with attention-weighted concatenation.
"""

import torch
import torch.nn as nn

from .image_model import create_image_model
from .biomarker_model import BiomarkerModel
from .eeg_model import SimpleEEGModel


class AttentionFusion(nn.Module):
    """
    Attention-based fusion mechanism for combining multi-modal features.
    Learns to weight the importance of each modality dynamically.
    """
    def __init__(self, feature_dims, hidden_dim=64):
        """
        Args:
            feature_dims: List of feature dimensions for each modality
            hidden_dim: Hidden dimension for attention computation
        """
        super(AttentionFusion, self).__init__()
        total_dim = sum(feature_dims)
        
        # Attention network
        self.attention = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(feature_dims)),
            nn.Softmax(dim=1)
        )
        
        # Project each modality to same dimension for weighted sum
        self.projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in feature_dims
        ])
        
    def forward(self, features_list):
        """
        Args:
            features_list: List of feature tensors from each modality
        Returns:
            Fused feature tensor
        """
        # Concatenate all features for attention computation
        concat_features = torch.cat(features_list, dim=1)
        
        # Compute attention weights
        attention_weights = self.attention(concat_features)  # (batch, num_modalities)
        
        # Project each modality
        projected = [proj(feat) for proj, feat in zip(self.projections, features_list)]
        
        # Weighted sum
        fused = sum(w.unsqueeze(-1) * p for w, p in zip(attention_weights.unbind(dim=1), projected))
        
        return fused, attention_weights


class FusionModel(nn.Module):
    """
    Trimodal Fusion Network for Comprehensive Stroke Classification.
    
    Combines three modalities:
    - Image (Brain MRI/CT): Vision Transformer extracts spatial features
    - Biomarker: MLP processes clinical parameters
    - EEG: 1D CNN analyzes brain electrical activity
    
    Uses late fusion with optional attention weighting.
    """
    
    def __init__(self, 
                 image_model_weights=None,
                 biomarker_model_weights=None,
                 eeg_model_weights=None,
                 biomarker_input_dim=29,
                 num_classes=3,
                 dropout_rate=0.3,
                 use_attention=True,
                 freeze_backbones=False):
        """
        Initialize the fusion model.

        Args:
            image_model_weights: Path to pretrained ViT weights (optional)
            biomarker_model_weights: Path to pretrained biomarker MLP weights (optional)
            eeg_model_weights: Path to pretrained EEG CNN weights (optional)
            biomarker_input_dim: Number of biomarker input features (default: 29)
            num_classes: Number of output classes (default: 3 for stroke types)
            dropout_rate: Dropout rate for fusion head
            use_attention: Whether to use attention-based fusion
            freeze_backbones: Whether to freeze pretrained backbone weights
        """
        super(FusionModel, self).__init__()
        
        self.use_attention = use_attention
        self.num_classes = num_classes
        
        # === Image Branch (Vision Transformer) ===
        self.image_model = create_image_model(num_classes=num_classes)
        if image_model_weights:
            try:
                state_dict = torch.load(image_model_weights, map_location='cpu')
                self.image_model.load_state_dict(state_dict)
                print("✓ Loaded pretrained image model weights")
            except Exception as e:
                print(f"⚠ Could not load image weights: {e}")
        
        # Get ViT feature dimension and replace head with identity
        self.vit_feature_dim = self.image_model.head.in_features  # Usually 768
        self.image_model.head = nn.Identity()
        
        # === Biomarker Branch (MLP) ===
        self.biomarker_model = BiomarkerModel(
            input_dim=biomarker_input_dim,
            hidden_dim1=128,
            hidden_dim2=64,
            output_dim=64  # Feature output, not classification
        )
        self.biomarker_feature_dim = 64
        
        if biomarker_model_weights:
            try:
                checkpoint = torch.load(biomarker_model_weights, map_location='cpu')
                # Load only compatible layers (input and hidden, not output)
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                # Partial loading - skip mismatched layers
                model_dict = self.biomarker_model.state_dict()
                pretrained_dict = {k: v for k, v in state_dict.items() 
                                 if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(pretrained_dict)
                self.biomarker_model.load_state_dict(model_dict, strict=False)
                print(f"✓ Loaded {len(pretrained_dict)}/{len(model_dict)} biomarker layers")
            except Exception as e:
                print(f"⚠ Could not load biomarker weights: {e}")
        
        # === EEG Branch (1D CNN) ===
        self.eeg_model = SimpleEEGModel(num_classes=64)  # Feature output
        self.eeg_feature_dim = 64
        
        if eeg_model_weights:
            try:
                state_dict = torch.load(eeg_model_weights, map_location='cpu')
                self.eeg_model.load_state_dict(state_dict, strict=False)
                print("✓ Loaded pretrained EEG model weights")
            except Exception as e:
                print(f"⚠ Could not load EEG weights: {e}")
        
        # === Freeze backbones if specified ===
        if freeze_backbones:
            for param in self.image_model.parameters():
                param.requires_grad = False
            for param in self.biomarker_model.parameters():
                param.requires_grad = False
            for param in self.eeg_model.parameters():
                param.requires_grad = False
            print("✓ Backbone weights frozen")
        
        # === Fusion Layer ===
        feature_dims = [self.vit_feature_dim, self.biomarker_feature_dim, self.eeg_feature_dim]
        total_features = sum(feature_dims)
        
        if use_attention:
            self.attention_fusion = AttentionFusion(feature_dims, hidden_dim=128)
            fusion_input_dim = 128  # Output of attention fusion
        else:
            self.attention_fusion = None
            fusion_input_dim = total_features
        
        # === Classification Head ===
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        print(f"✓ Fusion model initialized")
        print(f"  - Image features: {self.vit_feature_dim}")
        print(f"  - Biomarker features: {self.biomarker_feature_dim}")
        print(f"  - EEG features: {self.eeg_feature_dim}")
        print(f"  - Fusion method: {'Attention' if use_attention else 'Concatenation'}")
        print(f"  - Output classes: {num_classes}")

    def forward(self, image_data=None, biomarker_data=None, eeg_data=None):
        """
        Forward pass with flexible modality inputs.
        
        At least one modality must be provided. Missing modalities
        are handled by using zero features or learned defaults.

        Args:
            image_data: Brain scan tensor (batch, 3, 224, 224)
            biomarker_data: Clinical parameters tensor (batch, 29)
            eeg_data: EEG signal tensor (batch, 1, 256)

        Returns:
            output: Classification logits (batch, num_classes)
            attention_weights: Modality attention weights (if use_attention=True)
        """
        batch_size = None
        device = None
        
        # Determine batch size and device from available inputs
        for data in [image_data, biomarker_data, eeg_data]:
            if data is not None:
                batch_size = data.size(0)
                device = data.device
                break
        
        if batch_size is None:
            raise ValueError("At least one modality must be provided")
        
        # === Extract features from each modality ===
        features_list = []
        
        # Image features
        if image_data is not None:
            image_features = self.image_model(image_data)
        else:
            image_features = torch.zeros(batch_size, self.vit_feature_dim, device=device)
        features_list.append(image_features)
        
        # Biomarker features
        if biomarker_data is not None:
            biomarker_features = self.biomarker_model(biomarker_data)
        else:
            biomarker_features = torch.zeros(batch_size, self.biomarker_feature_dim, device=device)
        features_list.append(biomarker_features)
        
        # EEG features
        if eeg_data is not None:
            eeg_features = self.eeg_model(eeg_data)
        else:
            eeg_features = torch.zeros(batch_size, self.eeg_feature_dim, device=device)
        features_list.append(eeg_features)
        
        # === Fusion ===
        if self.use_attention and self.attention_fusion is not None:
            fused_features, attention_weights = self.attention_fusion(features_list)
        else:
            fused_features = torch.cat(features_list, dim=1)
            attention_weights = None
        
        # === Classification ===
        output = self.fusion_head(fused_features)
        
        if self.use_attention:
            return output, attention_weights
        return output
    
    def get_modality_contributions(self, image_data=None, biomarker_data=None, eeg_data=None):
        """
        Analyze the contribution of each modality to the final prediction.
        
        Returns:
            dict with attention weights and individual modality predictions
        """
        self.eval()
        with torch.no_grad():
            output, attention_weights = self.forward(image_data, biomarker_data, eeg_data)
            
            contributions = {
                'prediction': torch.softmax(output, dim=1),
                'attention_weights': attention_weights,
                'modality_names': ['Image (ViT)', 'Biomarker (MLP)', 'EEG (CNN)']
            }
            
            if attention_weights is not None:
                for i, name in enumerate(contributions['modality_names']):
                    contributions[f'{name}_weight'] = attention_weights[:, i].mean().item()
        
        return contributions


class BimodalFusionModel(FusionModel):
    """
    Simplified bimodal fusion using only Image + Biomarker.
    Useful when EEG data is not available.
    """
    def __init__(self, **kwargs):
        # Remove EEG-related parameters
        kwargs['eeg_model_weights'] = None
        super().__init__(**kwargs)
        
    def forward(self, image_data, biomarker_data):
        return super().forward(image_data=image_data, biomarker_data=biomarker_data, eeg_data=None)


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Fusion Model Architecture")
    print("=" * 60)
    
    # Initialize model without pretrained weights
    model = FusionModel(
        biomarker_input_dim=29,
        num_classes=3,
        use_attention=True
    )
    
    # Test with dummy inputs
    batch_size = 4
    image_input = torch.randn(batch_size, 3, 224, 224)
    biomarker_input = torch.randn(batch_size, 29)
    eeg_input = torch.randn(batch_size, 1, 256)
    
    print("\n--- Testing Full Trimodal Fusion ---")
    output, attention = model(image_input, biomarker_input, eeg_input)
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention.shape}")
    print(f"Attention weights: {attention[0].tolist()}")
    
    print("\n--- Testing with Missing Modalities ---")
    # Only image
    output1, _ = model(image_data=image_input)
    print(f"Image only - Output shape: {output1.shape}")
    
    # Only biomarker
    output2, _ = model(biomarker_data=biomarker_input)
    print(f"Biomarker only - Output shape: {output2.shape}")
    
    # Image + Biomarker
    output3, attn = model(image_data=image_input, biomarker_data=biomarker_input)
    print(f"Image + Biomarker - Output shape: {output3.shape}")
    
    print("\n--- Model Summary ---")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n✓ All tests passed!")
