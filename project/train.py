import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

class Trainer:
    def __init__(self, encoder, decoder, train_loader, val_loader, config, vocabulary):
        self.encoder = encoder
        self.decoder = decoder
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.vocabulary = vocabulary
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Modelleri device'a taşı
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        
        # Loss fonksiyonu
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.word2idx['<PAD>'])
        
        # Optimizer - sadece decoder ve attention eğitilecek
        self.decoder_optimizer = optim.Adam(
            params=filter(lambda p: p.requires_grad, decoder.parameters()),
            lr=config.learning_rate
        )
        
        # Encoder optimizer (fine-tuning için)
        self.encoder_optimizer = optim.Adam(
            params=filter(lambda p: p.requires_grad, encoder.parameters()),
            lr=config.learning_rate * 0.1
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.decoder_optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Loss geçmişi (grafik için)
        self.train_losses = []
        self.val_losses = []

        # Best model tracking
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.start_epoch = 1

        # Resume checkpoint varsa yükle
        resume_path = os.path.join(config.checkpoint_dir, 'last_checkpoint.pth')
        if os.path.exists(resume_path):
            print(f'Checkpoint bulundu, devam ediliyor: {resume_path}')
            ckpt = torch.load(resume_path, map_location=self.device, weights_only=False)
            self.encoder.load_state_dict(ckpt['encoder_state_dict'])
            self.decoder.load_state_dict(ckpt['decoder_state_dict'])
            self.decoder_optimizer.load_state_dict(ckpt['decoder_optimizer'])
            self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
            self.epochs_without_improvement = ckpt.get('epochs_without_improvement', 0)
            self.start_epoch = ckpt['epoch'] + 1
            print(f"Epoch {ckpt['epoch']}'dan devam ediliyor...")
        
    def train_epoch(self, epoch):
        self.encoder.train()
        self.decoder.train()
        
        total_loss = 0
        start_time = time.time()
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (images, captions, lengths, _) in enumerate(progress_bar):
            # Device'a taşı
            images = images.to(self.device)
            captions = captions.to(self.device)
            
            # Forward pass
            encoder_output = self.encoder(images)
            predictions, caps_sorted, decode_lengths, alphas, sort_idx = self.decoder(
                encoder_output, captions, lengths
            )
            
            # Target'ları hazırla
            targets = caps_sorted[:, 1:]
            
            # Pack padded sequence
            predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True)[0]
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]
            
            # Loss hesapla
            loss = self.criterion(predictions, targets)
            
            # Doubly stochastic attention regularization
            loss += self.config.alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean() if hasattr(self.config, 'alpha_c') else 0
            
            # Backward pass
            self.decoder_optimizer.zero_grad()
            if self.encoder_optimizer is not None:
                self.encoder_optimizer.zero_grad()
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.config.gradient_clip)
            
            # Optimizer step
            self.decoder_optimizer.step()
            if self.encoder_optimizer is not None:
                self.encoder_optimizer.step()
            
            # Tracking
            total_loss += loss.item()
            
            # Progress bar güncelle
            progress_bar.set_postfix({
                'loss': loss.item(),
                'avg_loss': total_loss / (batch_idx + 1)
            })
        
        avg_loss = total_loss / len(self.train_loader)
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch} - Train Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s')
        
        return avg_loss
    
    def validate(self, epoch):
        self.encoder.eval()
        self.decoder.eval()
        
        total_loss = 0
        
        with torch.no_grad():
            for images, captions, lengths, _ in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                captions = captions.to(self.device)
                
                # Forward pass
                encoder_output = self.encoder(images)
                predictions, caps_sorted, decode_lengths, alphas, sort_idx = self.decoder(
                    encoder_output, captions, lengths
                )
                
                # Target'ları hazırla
                targets = caps_sorted[:, 1:]
                
                # Pack padded sequence
                predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True)[0]
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]
                
                # Loss hesapla
                loss = self.criterion(predictions, targets)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        print(f'Epoch {epoch} - Val Loss: {avg_loss:.4f}')
        
        return avg_loss
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'decoder_optimizer': self.decoder_optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        checkpoint['best_val_loss'] = self.best_val_loss
        checkpoint['epochs_without_improvement'] = self.epochs_without_improvement

        last_path = os.path.join(self.config.checkpoint_dir, 'last_checkpoint.pth')
        torch.save(checkpoint, last_path)

        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f'Best model kaydedildi: {best_path}')
    
    def save_loss_plot(self, label=None, filename='loss_curve.png'):
        epochs = range(1, len(self.train_losses) + 1)
        _, ax = plt.subplots(figsize=(12, 6))
        ax.plot(epochs, self.train_losses, 'b-o', markersize=4, label='Train Loss')
        ax.plot(epochs, self.val_losses, 'r-o', markersize=4, label='Val Loss')

        # Her 5 epoch'a dikey çizgi
        for e in epochs:
            if e % 5 == 0:
                ax.axvline(x=e, color='gray', linestyle='--', alpha=0.4)
                ax.text(e, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else max(self.train_losses + self.val_losses),
                        f'E{e}', ha='center', va='bottom', fontsize=7, color='gray')

        title = label if label else f'Train / Val Loss (Epoch 1-{len(self.train_losses)})'
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(self.config.log_dir, filename)
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f'  [Grafik kaydedildi] {plot_path}')

    def train(self):
        print('Egitim basladi...')
        print(f'Device: {self.device}')
        print(f'Train batches: {len(self.train_loader)}')
        print(f'Val batches: {len(self.val_loader)}')
        
        for epoch in range(self.start_epoch, self.config.num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate(epoch)
            
            # Loss geçmişini kaydet
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.save_loss_plot(filename='loss_curve.png')

            # Her 5 epoch'ta snapshot
            if epoch % 5 == 0:
                self.save_loss_plot(
                    label=f'Train / Val Loss (Epoch 1-{epoch})',
                    filename=f'loss_curve_epoch_{epoch}.png'
                )

            # Learning rate scheduler
            self.scheduler.step(val_loss)

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Checkpoint kaydet (her epoch)
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Early stopping
            if self.epochs_without_improvement >= 10:
                print('Early stopping triggered')
                break
            
            # Belirli epoch'tan sonra encoder'ı fine-tune et
            if epoch == 10 and self.encoder_optimizer is None:
                print('Encoder fine-tuning baslatildi')
                self.encoder.unfreeze_encoder()
                self.encoder_optimizer = optim.Adam(
                    params=filter(lambda p: p.requires_grad, self.encoder.parameters()),
                    lr=self.config.learning_rate * 0.1
                )
        
        self.save_loss_plot(
            label=f'Final Train / Val Loss ({len(self.train_losses)} Epoch)',
            filename='loss_curve_final.png'
        )
        print('Egitim tamamlandi')