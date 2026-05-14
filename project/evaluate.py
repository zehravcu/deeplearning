import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.metrics import Metrics

class Evaluator:
    def __init__(self, encoder, decoder, data_loader, vocabulary, config):
        self.encoder = encoder
        self.decoder = decoder
        self.data_loader = data_loader
        self.vocabulary = vocabulary
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        
        self.metrics = Metrics()
        
    def evaluate(self):
        self.encoder.eval()
        self.decoder.eval()
        
        references = []
        hypotheses = []
        image_paths = []

        print('Degerlendirme baslatildi...')

        with torch.no_grad():
            for images, captions, _lengths, paths in tqdm(self.data_loader):
                images = images.to(self.device)

                encoder_output = self.encoder(images)

                start_token_idx = self.vocabulary.word2idx[self.config.start_token]
                end_token_idx = self.vocabulary.word2idx[self.config.end_token]

                sampled_ids = self.decoder.sample(
                    encoder_output,
                    start_token_idx,
                    end_token_idx,
                    max_length=self.config.max_seq_length
                )

                for i in range(images.size(0)):
                    ref_tokens = []
                    for idx in captions[i].cpu().numpy():
                        word = self.vocabulary.idx2word.get(idx, '<UNK>')
                        if word == '<END>':
                            break
                        if word not in ['<START>', '<PAD>', '<UNK>']:
                            ref_tokens.append(word)

                    hyp_tokens = []
                    for idx in sampled_ids[i].cpu().numpy():
                        word = self.vocabulary.idx2word.get(idx, '<UNK>')
                        if word == '<END>':
                            break
                        if word not in ['<START>', '<PAD>', '<UNK>']:
                            hyp_tokens.append(word)

                    references.append([ref_tokens])
                    hypotheses.append(hyp_tokens)
                    image_paths.append(paths[i])
        
        # Metrikleri hesapla
        print('\nMetrikler hesaplaniyor...')
        results = self.metrics.compute_all(references, hypotheses)

        print('\n=== SONUCLAR ===')
        for metric, value in results.items():
            print(f'{metric}: {value:.4f}')

        self.save_metrics_plot(results)
        self.save_eval_results(results, references, hypotheses, image_paths)

        return results

    def save_eval_results(self, results, references, hypotheses, image_paths):
        import random
        import textwrap
        from datetime import datetime
        from PIL import Image as PILImage

        n_samples = min(50, len(references))
        indices = random.sample(range(len(references)), n_samples)

        # eval_results.txt
        txt_path = os.path.join(self.config.log_dir, 'eval_results.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f'Degerlendirme Tarihi: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write('=' * 50 + '\n\n')
            f.write('=== METRIK SONUCLARI ===\n')
            for metric, value in results.items():
                f.write(f'{metric}: {value:.4f}\n')
            f.write(f'\n=== ORNEK TAHMINLER (rastgele {n_samples} ornek) ===\n')
            for rank, idx in enumerate(indices, 1):
                f.write(f'\n[{rank}] {os.path.basename(image_paths[idx])}\n')
                f.write(f'Gercek  : {" ".join(references[idx][0])}\n')
                f.write(f'Uretilen: {" ".join(hypotheses[idx])}\n')
        print(f'Eval sonuclari kaydedildi: {txt_path}')

        # Her örnek için PNG
        samples_dir = os.path.join(self.config.log_dir, 'report_samples')
        os.makedirs(samples_dir, exist_ok=True)

        for rank, idx in enumerate(indices, 1):
            img_path = image_paths[idx]
            ref_text = ' '.join(references[idx][0])
            hyp_text = ' '.join(hypotheses[idx]) if hypotheses[idx] else '(bos)'

            fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                                     gridspec_kw={'width_ratios': [1, 1.2]})

            # Sol: röntgen görüntüsü
            try:
                img = PILImage.open(img_path).convert('RGB')
                axes[0].imshow(img, cmap='gray')
            except Exception:
                axes[0].text(0.5, 0.5, 'Goruntu yuklenemedi', ha='center', va='center')
            axes[0].set_title(os.path.basename(img_path), fontsize=9)
            axes[0].axis('off')

            # Sağ: raporlar
            axes[1].axis('off')
            wrap = 60
            ref_wrapped = '\n'.join(textwrap.wrap(ref_text, wrap)) or '(bos)'
            hyp_wrapped = '\n'.join(textwrap.wrap(hyp_text, wrap)) or '(bos)'
            report_text = (
                f"GERCEK RAPOR:\n{ref_wrapped}\n\n"
                f"{'─'*50}\n\n"
                f"URETILEN RAPOR:\n{hyp_wrapped}"
            )
            axes[1].text(0.02, 0.95, report_text, transform=axes[1].transAxes,
                         fontsize=9, verticalalignment='top', family='monospace',
                         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

            fig.suptitle(f'Ornek {rank}', fontsize=11, fontweight='bold')
            plt.tight_layout()
            out_path = os.path.join(samples_dir, f'sample_{rank:02d}.png')
            plt.savefig(out_path, dpi=120, bbox_inches='tight')
            plt.close()

        print(f'{n_samples} gorsel rapor kaydedildi: {samples_dir}')

    def save_metrics_plot(self, results):
        metrics = list(results.keys())
        values = [results[m] for m in metrics]

        plt.figure(figsize=(10, 5))
        bars = plt.bar(metrics, values, color=['#4C72B0', '#4C72B0', '#4C72B0', '#4C72B0', '#DD8452', '#55A868'])
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        plt.title('Degerlendirme Metrikleri')
        plt.ylabel('Skor')
        plt.ylim(0, max(values) * 1.2 + 0.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot_path = os.path.join(self.config.log_dir, 'metrics.png')
        plt.savefig(plot_path)
        plt.close()
        print(f'Metrik grafigi kaydedildi: {plot_path}')
    
    def generate_report(self, image_path):
        # Tek bir görüntü için rapor üret
        from PIL import Image
        from torchvision import transforms
        
        self.encoder.eval()
        self.decoder.eval()
        
        # Görüntüyü yükle ve işle
        transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Encoder
            encoder_output = self.encoder(image)
            
            # Decoder
            start_token_idx = self.vocabulary.word2idx[self.config.start_token]
            end_token_idx = self.vocabulary.word2idx[self.config.end_token]
            
            sampled_ids = self.decoder.sample(
                encoder_output,
                start_token_idx,
                end_token_idx,
                max_length=self.config.max_seq_length
            )
            
            # Token'ları metne dönüştür
            report = self.vocabulary.decode(sampled_ids[0].cpu().numpy())
        
        return report