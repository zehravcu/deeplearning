import os
import torch
import argparse
from config import Config
from utils.vocabulary import Vocabulary
from utils.data_loader import create_data_loaders, IUXRayDataset
from models.encoder import ImageEncoder
from models.decoder import TextDecoder
from train import Trainer
from evaluate import Evaluator

def prepare_vocabulary(config):
    # Vocabulary hazırla veya yükle
    vocab_path = os.path.join(config.checkpoint_dir, 'vocabulary.pkl')
    
    if os.path.exists(vocab_path):
        print('Vocabulary yukleniyor...')
        vocabulary = Vocabulary.load(vocab_path)
    else:
        print('Vocabulary olusturuluyor...')
        import csv

        all_captions = []
        reports_path = os.path.join(config.reports_dir, 'indiana_reports.csv')
        with open(reports_path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                findings = row.get('findings', '') or ''
                impression = row.get('impression', '') or ''
                caption = (findings + ' ' + impression).strip()
                if caption:
                    all_captions.append(caption)
        
        # Vocabulary oluştur
        vocabulary = Vocabulary(min_freq=config.min_word_freq)
        special_tokens = [config.pad_token, config.start_token, config.end_token, config.unk_token]
        vocabulary.build_vocabulary(all_captions, special_tokens)
        
        # Kaydet
        vocabulary.save(vocab_path)
        print(f'Vocabulary olusturuldu: {len(vocabulary)} kelime')
    
    return vocabulary

def train_model(config):
    print('=== EGITIM MODU ===')
    
    # Dizinleri oluştur
    config.create_dirs()
    
    # Vocabulary hazırla
    vocabulary = prepare_vocabulary(config)
    
    # Data loader'ları oluştur
    print('Veri yukleniyor...')
    train_loader, val_loader = create_data_loaders(config, vocabulary)
    
    # Modelleri oluştur
    print('Modeller olusturuluyor...')
    encoder = ImageEncoder(encoded_size=config.encoder_dim)
    decoder = TextDecoder(
        attention_dim=config.attention_dim,
        embedding_dim=config.embedding_dim,
        decoder_dim=config.decoder_dim,
        vocab_size=len(vocabulary),
        encoder_dim=config.encoder_dim,
        dropout=config.dropout
    )
    
    print(f'Encoder parametreleri: {sum(p.numel() for p in encoder.parameters())}')
    print(f'Decoder parametreleri: {sum(p.numel() for p in decoder.parameters())}')
    
    # Trainer oluştur ve eğit
    trainer = Trainer(encoder, decoder, train_loader, val_loader, config, vocabulary)
    trainer.train()

def evaluate_model(config):
    print('=== DEGERLENDIRME MODU ===')
    
    # Vocabulary yükle
    vocab_path = os.path.join(config.checkpoint_dir, 'vocabulary.pkl')
    if not os.path.exists(vocab_path):
        print('Hata: Vocabulary bulunamadi')
        return
    
    vocabulary = Vocabulary.load(vocab_path)
    
    # Data loader
    print('Veri yukleniyor...')
    _, val_loader = create_data_loaders(config, vocabulary)
    
    # Modelleri oluştur
    encoder = ImageEncoder(encoded_size=config.encoder_dim)
    decoder = TextDecoder(
        attention_dim=config.attention_dim,
        embedding_dim=config.embedding_dim,
        decoder_dim=config.decoder_dim,
        vocab_size=len(vocabulary),
        encoder_dim=config.encoder_dim,
        dropout=config.dropout
    )
    
    # Checkpoint yükle
    checkpoint_path = config.best_model_path
    if not os.path.exists(checkpoint_path):
        print(f'Hata: Checkpoint bulunamadi: {checkpoint_path}')
        return
    
    print(f'Checkpoint yukleniyor: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    # Evaluator oluştur
    evaluator = Evaluator(encoder, decoder, val_loader, vocabulary, config)
    evaluator.evaluate()

def generate_single_report(config, image_path):
    print('=== RAPOR URETME MODU ===')
    
    # Vocabulary yükle
    vocab_path = os.path.join(config.checkpoint_dir, 'vocabulary.pkl')
    vocabulary = Vocabulary.load(vocab_path)
    
    # Modelleri oluştur
    encoder = ImageEncoder(encoded_size=config.encoder_dim)
    decoder = TextDecoder(
        attention_dim=config.attention_dim,
        embedding_dim=config.embedding_dim,
        decoder_dim=config.decoder_dim,
        vocab_size=len(vocabulary),
        encoder_dim=config.encoder_dim,
        dropout=config.dropout
    )
    
    # Checkpoint yükle
    checkpoint = torch.load(config.best_model_path, map_location='cpu', weights_only=False)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    # Evaluator oluştur
    evaluator = Evaluator(encoder, decoder, None, vocabulary, config)
    
    # Rapor üret
    report = evaluator.generate_report(image_path)
    
    print(f'\nGoruntu: {image_path}')
    print(f'Rapor: {report}')

    reports_file = os.path.join(config.log_dir, 'generated_reports.txt')
    from datetime import datetime
    with open(reports_file, 'a', encoding='utf-8') as f:
        f.write(f'\n[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n')
        f.write(f'Goruntu: {image_path}\n')
        f.write(f'Rapor  : {report}\n')
        f.write('-' * 60 + '\n')
    print(f'Rapor kaydedildi: {reports_file}')

def main():
    parser = argparse.ArgumentParser(description='Tibbi Goruntu Raporlama')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'eval', 'generate'],
                       help='Calistirma modu: train, eval, generate')
    parser.add_argument('--image_path', type=str, default=None,
                       help='Rapor uretilecek goruntu yolu (generate modu icin)')
    
    args = parser.parse_args()
    
    # Config oluştur
    config = Config()
    
    # Mod seçimi
    if args.mode == 'train':
        train_model(config)
    elif args.mode == 'eval':
        evaluate_model(config)
    elif args.mode == 'generate':
        if args.image_path is None:
            print('Hata: --image_path parametresi gerekli')
            return
        generate_single_report(config, args.image_path)

if __name__ == '__main__':
    main()