import torch
import torch.nn as nn
from models.attention import CrossAttention

class TextDecoder(nn.Module):
    def __init__(self, attention_dim, embedding_dim, decoder_dim, vocab_size, encoder_dim, dropout=0.5):
        super(TextDecoder, self).__init__()
        
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embedding_dim = embedding_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        
        # Attention mekanizması
        self.attention = CrossAttention(encoder_dim, decoder_dim, attention_dim)
        
        # Embedding katmanı
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # LSTM decoder
        self.lstm_cell = nn.LSTMCell(embedding_dim + encoder_dim, decoder_dim)
        
        # Hidden state başlatma
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        
        # Çıktı katmanı
        self.fc = nn.Linear(decoder_dim, vocab_size)
        
        # Gate mekanizması (sentinel)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        
    def init_hidden_state(self, encoder_output):
        # encoder_output: (batch_size, num_pixels, encoder_dim)
        
        # Global average pooling
        mean_encoder_output = encoder_output.mean(dim=1)
        # mean_encoder_output: (batch_size, encoder_dim)
        
        # Hidden ve cell state'leri başlat
        h = self.init_h(mean_encoder_output)
        c = self.init_c(mean_encoder_output)
        
        return h, c
    
    def forward(self, encoder_output, captions, caption_lengths):
        # encoder_output: (batch_size, num_pixels, encoder_dim)
        # captions: (batch_size, max_caption_length)
        # caption_lengths: list of lengths
        
        batch_size = encoder_output.size(0)
        num_pixels = encoder_output.size(1)
        vocab_size = self.vocab_size
        
        # Sıralama
        caption_lengths, sort_idx = torch.sort(torch.LongTensor(caption_lengths), descending=True)
        encoder_output = encoder_output[sort_idx]
        captions = captions[sort_idx]
        
        # Embedding
        embeddings = self.embedding(captions)
        # embeddings: (batch_size, max_caption_length, embedding_dim)
        
        # Hidden state'leri başlat
        h, c = self.init_hidden_state(encoder_output)
        
        # Maksimum decode uzunluğu
        decode_lengths = (caption_lengths - 1).tolist()
        
        # Çıktılar için tensor
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(encoder_output.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_output.device)
        
        # Her zaman adımında decode et
        for t in range(max(decode_lengths)):
            # Bu adımda kaç örnek aktif
            batch_size_t = sum([l > t for l in decode_lengths])
            
            # Attention uygula
            context, alpha = self.attention(encoder_output[:batch_size_t], h[:batch_size_t])
            
            # Gate mekanizması
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            context = gate * context
            
            # LSTM girdisi
            lstm_input = torch.cat([embeddings[:batch_size_t, t, :], context], dim=1)
            
            # LSTM adımı
            h, c = self.lstm_cell(lstm_input, (h[:batch_size_t], c[:batch_size_t]))
            
            # Tahmin
            preds = self.fc(self.dropout(h))
            
            # Kaydet
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
        
        return predictions, captions, decode_lengths, alphas, sort_idx
    
    def sample(self, encoder_output, start_token_idx, end_token_idx, max_length=100):
        # Beam search veya greedy sampling
        batch_size = encoder_output.size(0)
        
        # Hidden state'leri başlat
        h, c = self.init_hidden_state(encoder_output)
        
        # İlk token
        current_token = torch.LongTensor([start_token_idx] * batch_size).to(encoder_output.device)
        
        sampled_ids = []
        
        for t in range(max_length):
            # Embedding
            embeddings = self.embedding(current_token)
            
            # Attention
            context, alpha = self.attention(encoder_output, h)
            
            # Gate
            gate = self.sigmoid(self.f_beta(h))
            context = gate * context
            
            # LSTM
            lstm_input = torch.cat([embeddings, context], dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            
            # Tahmin
            preds = self.fc(h)
            
            # Greedy sampling
            predicted = preds.argmax(dim=1)
            sampled_ids.append(predicted)
            
            # Bir sonraki token
            current_token = predicted
            
            # End token kontrolü
            if (predicted == end_token_idx).all():
                break
        
        sampled_ids = torch.stack(sampled_ids, dim=1)
        return sampled_ids