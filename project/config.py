import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    def __init__(self):
        # Veri yolları
        self.data_dir = os.path.join(BASE_DIR, 'data', 'iu_xray')
        self.images_dir = os.path.join(self.data_dir, 'images', 'images_normalized')
        self.reports_dir = os.path.join(self.data_dir, 'reports')

        # Model parametreleri
        self.image_size = 224
        self.encoder_dim = 1024
        self.attention_dim = 512
        self.decoder_dim = 512
        self.embedding_dim = 512
        self.dropout = 0.5

        # Eğitim parametreleri
        self.batch_size = 16
        self.num_epochs = 75
        self.learning_rate = 0.0001
        self.teacher_forcing_ratio = 0.8
        self.gradient_clip = 5.0

        # Token ayarları
        self.max_seq_length = 100
        self.min_word_freq = 3

        # Checkpoint ve log
        self.checkpoint_dir = os.path.join(BASE_DIR, 'checkpoints')
        self.log_dir = os.path.join(BASE_DIR, 'logs')
        self.best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        
        # GPU ayarları
        self.device = 'cuda'
        self.num_workers = 4
        
        # Tokenlar
        self.pad_token = '<PAD>'
        self.start_token = '<START>'
        self.end_token = '<END>'
        self.unk_token = '<UNK>'
        
    def create_dirs(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)