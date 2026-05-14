import torch
import torch.nn as nn
from torchvision import models

class ImageEncoder(nn.Module):
    def __init__(self, encoded_size=1024):
        super(ImageEncoder, self).__init__()
        
        # DenseNet-121 önceden eğitilmiş ağırlıklarla
        densenet = models.densenet121(pretrained=True)
        
        # Son katmanı çıkar, sadece özellik çıkarıcıyı kullan
        modules = list(densenet.children())[:-1]
        self.densenet = nn.Sequential(*modules)
        
        # Adaptif pooling ekle
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # DenseNet-121'in çıktı boyutu 1024
        # Bu boyutu istediğimiz encoder_dim'e dönüştürebiliriz
        self.feature_projection = nn.Linear(1024, encoded_size)
        
        # Encoder kısmını dondur (transfer learning)
        self._freeze_encoder()
        
    def _freeze_encoder(self):
        # DenseNet parametrelerini dondur
        for param in self.densenet.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        # Fine-tuning için encoder'ı çöz
        for param in self.densenet.parameters():
            param.requires_grad = True
    
    def forward(self, images):
        # images: (batch_size, 3, 224, 224)
        
        # DenseNet'ten özellik çıkar
        features = self.densenet(images)
        # features: (batch_size, 1024, 7, 7)
        
        # Adaptive pooling uygula
        features = self.adaptive_pool(features)
        # features: (batch_size, 1024, 7, 7)
        
        # Spatial boyutları düzleştir
        batch_size = features.size(0)
        features = features.view(batch_size, 1024, -1)
        # features: (batch_size, 1024, 49)
        
        # Permute et: (batch_size, num_pixels, feature_dim)
        features = features.permute(0, 2, 1)
        # features: (batch_size, 49, 1024)
        
        # Projeksiyon uygula
        features = self.feature_projection(features)
        # features: (batch_size, 49, encoded_size)
        
        return features