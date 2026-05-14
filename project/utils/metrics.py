import torch
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge

class Metrics:
    def __init__(self):
        self.rouge = Rouge()
        
    def calculate_bleu(self, references, hypotheses):
        # BLEU-1, BLEU-2, BLEU-3, BLEU-4
        bleu_scores = []
        
        for n in range(1, 5):
            weights = tuple([1.0/n] * n + [0] * (4-n))
            score = corpus_bleu(references, hypotheses, weights=weights)
            bleu_scores.append(score)
        
        return bleu_scores
    
    def calculate_meteor(self, references, hypotheses):
        scores = []
        for ref, hyp in zip(references, hypotheses):
            if not hyp or not ref[0]:
                continue
            score = meteor_score([ref[0]], hyp)
            scores.append(score)

        return np.mean(scores) if scores else 0.0
    
    def calculate_rouge(self, references, hypotheses):
        # ROUGE-L hesapla
        rouge_scores = []
        
        for ref, hyp in zip(references, hypotheses):
            ref_str = ' '.join(ref[0])
            hyp_str = ' '.join(hyp)
            
            if len(ref_str) > 0 and len(hyp_str) > 0:
                try:
                    score = self.rouge.get_scores(hyp_str, ref_str)[0]['rouge-l']['f']
                    rouge_scores.append(score)
                except:
                    continue
        
        return np.mean(rouge_scores) if rouge_scores else 0.0
    
    def compute_all(self, references, hypotheses):
        # Tüm metrikleri hesapla
        bleu_scores = self.calculate_bleu(references, hypotheses)
        
        results = {
            'BLEU-1': bleu_scores[0],
            'BLEU-2': bleu_scores[1],
            'BLEU-3': bleu_scores[2],
            'BLEU-4': bleu_scores[3],
        }
        
        try:
            results['METEOR'] = self.calculate_meteor(references, hypotheses)
        except:
            results['METEOR'] = 0.0
        
        try:
            results['ROUGE-L'] = self.calculate_rouge(references, hypotheses)
        except:
            results['ROUGE-L'] = 0.0
        
        return results

def accuracy(predictions, targets, k=1):
    # Top-k accuracy
    batch_size = targets.size(0)
    _, pred = predictions.topk(k, 1, True, True)
    correct = pred.eq(targets.view(-1, 1).expand_as(pred))
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() / batch_size