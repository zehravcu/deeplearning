import os
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

class IUXRayDataset(Dataset):
    def __init__(self, images_dir, reports_dir, vocabulary, config, mode='train'):
        self.images_dir = images_dir
        self.reports_dir = reports_dir
        self.vocabulary = vocabulary
        self.config = config
        self.mode = mode
        self.samples = []
        
        # Veri çiftlerini yükle
        self._load_data()
        
        # Görüntü ön işleme
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((config.image_size, config.image_size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def _load_data(self):
        # uid -> filename eşlemesini oluştur
        projections_path = os.path.join(self.reports_dir, 'indiana_projections.csv')
        uid_to_files = {}
        with open(projections_path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                uid = row['uid']
                uid_to_files.setdefault(uid, []).append(row['filename'])

        reports_path = os.path.join(self.reports_dir, 'indiana_reports.csv')
        with open(reports_path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                findings = row.get('findings', '') or ''
                impression = row.get('impression', '') or ''
                caption = (findings + ' ' + impression).strip()
                if not caption:
                    continue

                uid = row['uid']
                for filename in uid_to_files.get(uid, []):
                    image_path = os.path.join(self.images_dir, filename)
                    if os.path.exists(image_path):
                        self.samples.append({
                            'image_path': image_path,
                            'caption': caption
                        })

        print(f"{self.mode} modu: {len(self.samples)} ornek yuklendi")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Görüntüyü yükle ve işle
        image = Image.open(sample['image_path']).convert('RGB')
        image = self.transform(image)
        
        # Metni tokenize et
        caption = sample['caption']
        tokens = [self.config.start_token] + caption.lower().split() + [self.config.end_token]
        
        # İndekslere çevir
        caption_indices = [self.vocabulary.word2idx.get(token, self.vocabulary.word2idx['<UNK>']) 
                          for token in tokens]
        
        # Uzunluk sınırla
        if len(caption_indices) > self.config.max_seq_length:
            caption_indices = caption_indices[:self.config.max_seq_length]
        
        # Padding ekle
        caption_length = len(caption_indices)
        padded_caption = caption_indices + [self.vocabulary.word2idx['<PAD>']] * (self.config.max_seq_length - len(caption_indices))
        
        return {
            'image': image,
            'caption': torch.LongTensor(padded_caption),
            'caption_length': caption_length,
            'image_path': sample['image_path']
        }

def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    captions = torch.stack([item['caption'] for item in batch])
    lengths = [item['caption_length'] for item in batch]
    image_paths = [item['image_path'] for item in batch]

    return images, captions, lengths, image_paths

def create_data_loaders(config, vocabulary):
    train_dataset = IUXRayDataset(
        config.images_dir,
        config.reports_dir,
        vocabulary,
        config,
        mode='train'
    )
    
    # Veriyi train/val olarak böl
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader