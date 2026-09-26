from __future__ import annotations
import numpy as np
import torch
from PIL import Image
import matplotlib.cm as cm

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.h1 = target_layer.register_forward_hook(self._forward)
        self.h2 = target_layer.register_full_backward_hook(self._backward)

    def _forward(self, module, inputs, output):
        self.activations = output

    def _backward(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        logits[0, class_idx].backward()
        a = self.activations[0].detach()
        g = self.gradients[0].detach()
        weights = g.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * a).sum(0))
        cam -= cam.min()
        cam /= cam.max().clamp_min(1e-8)
        return cam.cpu().numpy(), class_idx

    def close(self):
        self.h1.remove(); self.h2.remove()

def overlay_cam(image: Image.Image, cam: np.ndarray, alpha: float = 0.38) -> Image.Image:
    image = image.convert("RGB")
    heat = Image.fromarray(np.uint8(cam * 255)).resize(image.size, Image.Resampling.BILINEAR)
    heat = np.asarray(heat, dtype=np.float32) / 255.0
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    colored = cm.get_cmap("jet")(heat)[..., :3]
    out = np.clip((1-alpha) * rgb + alpha * colored, 0, 1)
    return Image.fromarray(np.uint8(out * 255))
