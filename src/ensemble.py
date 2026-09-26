from __future__ import annotations
import torch

@torch.inference_mode()
def ensemble_predict(models, x):
    """Return mean probability, variance, predictive entropy and mutual information."""
    probs=[]
    for model in models:
        model.eval()
        probs.append(torch.softmax(model(x),dim=1))
    samples=torch.stack(probs)
    mean=samples.mean(0)
    variance=samples.var(0,unbiased=False)
    p=mean.clamp_min(1e-8)
    entropy=-(p*p.log()).sum(1)
    sp=samples.clamp_min(1e-8)
    sample_entropy=-(sp*sp.log()).sum(2).mean(0)
    mi=entropy-sample_entropy
    return mean,variance,entropy,mi
