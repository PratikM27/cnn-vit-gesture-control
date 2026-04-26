"""
cnn_model.py — CNN Gesture Classifier (Baseline)
==================================================
Custom CNN and MobileNetV2-based classifiers for hand gesture recognition.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


class GestureCNN(nn.Module):
    """
    Custom 4-layer CNN for gesture classification.
    
    Architecture:
        Conv2D(3→32)  → BN → ReLU → MaxPool
        Conv2D(32→64) → BN → ReLU → MaxPool
        Conv2D(64→128)→ BN → ReLU → MaxPool
        Conv2D(128→256)→ BN → ReLU → MaxPool
        AdaptiveAvgPool → Flatten
        FC(256→512) → ReLU → Dropout
        FC(512→num_classes)
    
    Input: (B, 3, 128, 128)
    Output: (B, num_classes)
    """
    
    def __init__(self, num_classes=7, dropout=0.5):
        super(GestureCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1: 128→64
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 64→32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 32→16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4: 16→8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Kaiming initialization for conv and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


class MobileNetV2Classifier(nn.Module):
    """
    MobileNetV2-based gesture classifier (pretrained, fine-tuned).
    
    Uses torchvision's MobileNetV2 with a custom classification head.
    Input: (B, 3, 128, 128)
    Output: (B, num_classes)
    """
    
    def __init__(self, num_classes=7, pretrained=True, dropout=0.5):
        super(MobileNetV2Classifier, self).__init__()
        
        # Load pretrained MobileNetV2
        weights = tv_models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = tv_models.mobilenet_v2(weights=weights)
        
        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze all layers except classifier."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all layers."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def build_cnn_model(model_name="custom_cnn", num_classes=7, dropout=0.5):
    """
    Factory function to build CNN model.
    
    Args:
        model_name: "custom_cnn" or "mobilenetv2"
        num_classes: Number of gesture classes
        dropout: Dropout rate
    
    Returns:
        nn.Module: The CNN model
    """
    if model_name == "custom_cnn":
        return GestureCNN(num_classes=num_classes, dropout=dropout)
    elif model_name == "mobilenetv2":
        return MobileNetV2Classifier(num_classes=num_classes, dropout=dropout)
    else:
        raise ValueError(f"Unknown CNN model: {model_name}. Use 'custom_cnn' or 'mobilenetv2'.")


if __name__ == "__main__":
    # Quick test
    model = GestureCNN(num_classes=7)
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"GestureCNN: input={x.shape} → output={out.shape}")
    print(f"Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    model2 = MobileNetV2Classifier(num_classes=7, pretrained=False)
    out2 = model2(x)
    total_params2 = sum(p.numel() for p in model2.parameters())
    print(f"\nMobileNetV2: input={x.shape} → output={out2.shape}")
    print(f"Parameters: {total_params2:,} ({total_params2/1e6:.2f}M)")
