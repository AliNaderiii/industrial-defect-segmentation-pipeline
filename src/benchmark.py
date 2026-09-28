"""Benchmark for industrial defect segmentation"""
import torch, time
from models import get_model, count_parameters
import numpy as np

def benchmark(model_name='unet', encoder='resnet18', img_size=256, num_runs=50):
    device = torch.device('cpu')
    model = get_model(model_name, num_classes=2, encoder=encoder)
    model.eval()
    params = count_parameters(model)
    x = torch.randn(1, 3, img_size, img_size)
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)
    times = []
    for _ in range(num_runs):
        s = time.time()
        with torch.no_grad():
            _ = model(x)
        times.append((time.time()-s)*1000)
    print(f"{model_name} {encoder} {img_size}px: {params:.1f}M params, {np.mean(times):.1f}±{np.std(times):.1f} ms")
    return params, np.mean(times)

if __name__ == "__main__":
    for m,e in [('unet','resnet18'), ('deeplabv3plus','resnet50')]:
        benchmark(m,e,128,20)
