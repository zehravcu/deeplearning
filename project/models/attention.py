import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(CrossAttention, self).__init__()
        
        # Encoder özelliklerini attention space'e dönüştür
        self.encoder_attention = nn.Linear(encoder_dim, attention_dim)
        
        # Decoder hidden state'i attention space'e dönüştür
        self.decoder_attention = nn.Linear(decoder_dim, attention_dim)
        
        # Attention skorları için son katman
        self.full_attention = nn.Linear(attention_dim, 1)
        
        # Activation
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, encoder_output, decoder_hidden):
        # encoder_output: (batch_size, num_pixels, encoder_dim)
        # decoder_hidden: (batch_size, decoder_dim)
        
        # Encoder çıktısını attention space'e dönüştür
        att1 = self.encoder_attention(encoder_output)
        # att1: (batch_size, num_pixels, attention_dim)
        
        # Decoder hidden'ı attention space'e dönüştür
        att2 = self.decoder_attention(decoder_hidden)
        # att2: (batch_size, attention_dim)
        
        # Broadcast için boyut ekle
        att2 = att2.unsqueeze(1)
        # att2: (batch_size, 1, attention_dim)
        
        # İkisini topla ve aktivasyon uygula
        att_combined = self.relu(att1 + att2)
        # att_combined: (batch_size, num_pixels, attention_dim)
        
        # Attention skorlarını hesapla
        att_scores = self.full_attention(att_combined).squeeze(2)
        # att_scores: (batch_size, num_pixels)
        
        # Softmax ile normalize et
        alpha = self.softmax(att_scores)
        # alpha: (batch_size, num_pixels)
        
        # Attention ağırlıklarını encoder çıktısına uygula
        alpha = alpha.unsqueeze(2)
        # alpha: (batch_size, num_pixels, 1)
        
        # Weighted sum
        context = (encoder_output * alpha).sum(dim=1)
        # context: (batch_size, encoder_dim)
        
        return context, alpha.squeeze(2)