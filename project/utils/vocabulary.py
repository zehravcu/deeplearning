from collections import Counter
import pickle

class Vocabulary:
    def __init__(self, min_freq=3):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        
    def build_vocabulary(self, captions, special_tokens):
        # Tüm kelimelerin frekansını say
        for caption in captions:
            tokens = caption.lower().split()
            self.word_freq.update(tokens)
        
        # Özel tokenları ekle
        self.word2idx = {token: idx for idx, token in enumerate(special_tokens)}
        
        # Minimum frekanstan fazla geçen kelimeleri ekle
        idx = len(special_tokens)
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq:
                self.word2idx[word] = idx
                idx += 1
        
        # Ters mapping oluştur
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
    def encode(self, text, max_length=None):
        tokens = text.lower().split()
        indices = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
        
        if max_length:
            if len(indices) < max_length:
                indices += [self.word2idx['<PAD>']] * (max_length - len(indices))
            else:
                indices = indices[:max_length]
                
        return indices
    
    def decode(self, indices, skip_special=True):
        words = []
        special = ['<PAD>', '<START>', '<END>', '<UNK>']
        
        for idx in indices:
            word = self.idx2word.get(idx, '<UNK>')
            if skip_special and word in special:
                continue
            if word == '<END>':
                break
            words.append(word)
            
        return ' '.join(words)
    
    def __len__(self):
        return len(self.word2idx)
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)