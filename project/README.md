# Tibbi Goruntu Raporlama Projesi

Gogus rontgeni goruntulerinden otomatik radyoloji raporu ureten derin ogrenme projesi.

## Proje Yapisi

```
medical_image_captioning/
├── config.py                 # Konfigurasyon ayarlari
├── main.py                   # Ana calistirma scripti
├── train.py                  # Egitim dongusu
├── evaluate.py               # Degerlendirme ve rapor uretme
├── requirements.txt          # Gerekli kutuphaneler
│
├── models/
│   ├── encoder.py           # DenseNet-121 encoder
│   ├── attention.py         # Cross-attention mekanizmasi
│   └── decoder.py           # LSTM decoder
│
├── utils/
│   ├── vocabulary.py        # Kelime haznesi
│   ├── data_loader.py       # Veri yukleme
│   └── metrics.py           # Degerlendirme metrikleri
│
├── data/
│   └── iu_xray/
│       ├── images/          # Rontgen goruntulerini buraya
│       └── reports/         # XML raporlarini buraya
│
├── checkpoints/             # Egitilmis modeller
└── logs/                    # Egitim loglari
```

## Kurulum

1. Gerekli kutuphaneleri yukleyin:
```bash
pip install -r requirements.txt
```

2. NLTK verilerini indirin:
```python
import nltk
nltk.download('wordnet')
nltk.download('punkt')
```

3. IU X-Ray veri setini indirin:
- Kaggle'dan IU X-Ray veri setini indirin
- Goruntuler: data/iu_xray/images/
- Raporlar: data/iu_xray/reports/

## Kullanim

### 1. Model Egitimi

```bash
python main.py --mode train
```

Egitim parametreleri config.py dosyasindan degistirilebilir.

### 2. Model Degerlendirme

```bash
python main.py --mode eval
```

BLEU, METEOR, ROUGE-L metriklerini hesaplar.

### 3. Tek Goruntu icin Rapor Uretme

```bash
python main.py --mode generate --image_path path/to/image.png
```

## Model Mimarisi

### Encoder
- DenseNet-121 (ImageNet agirliklarla onceden egitilmis)
- Transfer learning ile ozellik cikarimi
- Cikti: 49 piksel x 1024 boyutunda ozellik haritasi

### Attention
- Cross-attention mekanizmasi
- Decoder'in her adiminda goruntuye odaklanma
- Dinamik context vector uretimi

### Decoder
- LSTM tabanli metin uretici
- Embedding boyutu: 512
- Hidden state boyutu: 512
- Teacher forcing ile egitim

## Egitim Detaylari

- Batch size: 16
- Learning rate: 0.0001
- Optimizer: Adam
- Max epochs: 50
- Early stopping: 10 epoch
- Gradient clipping: 5.0

## Transfer Learning Stratejisi

1. Ilk 10 epoch: Sadece decoder ve attention egitilir
2. 10. epoch'tan sonra: Encoder da fine-tune edilir
3. Encoder learning rate: decoder'in 1/10'u

## RTX 3050 icin Optimizasyon Onerileri

config.py dosyasinda:
- batch_size = 8 veya 16 (bellek durumuna gore)
- num_workers = 2 veya 4
- Mixed precision training eklenebilir

## Metrikler

- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- METEOR
- ROUGE-L

## Ornek Cikti

```
Goruntu: patient_001.png
Rapor: the heart is normal in size and shape there is no evidence of 
pulmonary edema or infiltrate the lungs are clear no pleural effusion 
no pneumothorax is identified
```

## Notlar

- Ilk egitim 2-3 saat surebilir (RTX 3050)
- Vocabulary otomatik olusturulur
- Best model checkpoints/ klasorune kaydedilir
- Egitim sirasinda validation loss izlenir