# Requires: torch, torchvision, pillow, numpy, matplotlib
# pip uninstall torch torchvision -y
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# pip install pillow matplotlib numpy



import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

# -------------------------
# Utility: register hooks to capture activations and gradients
# -------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        """
        model: a torch.nn.Module (evaluation mode recommended)
        target_layer: the layer object (not name) whose activations we want, e.g. model.features[-1]
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        # register forward hook
        self.fwd_handle = self.target_layer.register_forward_hook(self._save_activations)
        # register backward hook
        self.bwd_handle = self.target_layer.register_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        # output shape: (B, C, H, W)
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        # grad_output is a tuple (grad,)
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, target_class=None):
        """
        input_tensor: 1xCxHxW tensor on same device as model
        target_class: integer class index to compute Grad-CAM for. If None, uses predicted class.
        Returns: cam (H_orig, W_orig) normalized to [0,1]
        """
        self.model.zero_grad()
        out = self.model(input_tensor)  # logits or scores
        if out.dim() == 1 or out.shape[0] == 1:
            out = out.view(1, -1)

        if target_class is None:
            # predicted class
            pred_prob = F.softmax(out, dim=1)
            target_class = int(pred_prob.argmax(dim=1).item())

        score = out[0, target_class]
        score.backward(retain_graph=True)

        # activations: (1, C, H, W), gradients: (1, C, H, W)
        activations = self.activations[0]   # (C, H, W)
        gradients = self.gradients[0]       # (C, H, W)

        # global average pooling of gradients -> weights (C,)
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # weighted sum of activations
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)  # (H, W)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        # relu
        cam = torch.relu(cam)

        # normalize to [0,1]
        cam -= cam.min()
        if cam.max() != 0:
            cam /= cam.max()

        cam_np = cam.cpu().numpy()  # (H, W), values in [0,1]
        return cam_np

    def close(self):
        # remove hooks
        self.fwd_handle.remove()
        self.bwd_handle.remove()


# -------------------------
# Image preprocessing / postprocessing helpers
# -------------------------
def preprocess_image(image_path, input_size=(224, 224)):
    """
    Loads image and returns:
      - original PIL image
      - preprocessed tensor (1,C,H,W) ready for model
    """
    img = Image.open(image_path).convert('RGB')
    orig_size = img.size  # (W, H)

    preprocess = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats (change if needed)
                             std=[0.229, 0.224, 0.225])
    ])
    tensor = preprocess(img).unsqueeze(0)  # 1xCxHxW
    return img, tensor, orig_size

def resize_cam_to_image(cam, orig_size):
    """
    cam: numpy array (H_cam, W_cam) in [0,1]
    orig_size: (W_orig, H_orig)
    returns: resized cam in [0,1] shape (H_orig, W_orig)
    """
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(orig_size, resample=Image.BILINEAR)
    cam_resized = np.array(cam_img).astype(np.float32) / 255.0
    return cam_resized

def apply_colormap_on_image(original_pil, cam_resized, colormap=cm.jet, alpha=0.4):
    """
    original_pil: PIL image (RGB)
    cam_resized: numpy array (H, W) in [0,1]
    alpha: blending factor for the heatmap
    returns: overlay_pil (PIL RGB)
    """
    # normalize original image to [0,1]
    orig = np.asarray(original_pil).astype(np.float32) / 255.0
    # apply colormap (returns RGBA)
    colored_cam = colormap(cam_resized)[:, :, :3]  # take RGB
    # blend
    overlay = (1 - alpha) * orig + alpha * colored_cam
    overlay = np.clip(overlay, 0, 1)
    overlay_uint8 = np.uint8(overlay * 255)
    return Image.fromarray(overlay_uint8)

def save_colorbar(colormap=cm.jet, filename='heatmap_colorbar.png', vmin=0.0, vmax=1.0):
    # create a fake ScalarMappable for colorbar
    sm = cm.ScalarMappable(cmap=colormap)
    sm.set_array(np.linspace(vmin, vmax, 256))
    plt.figure(figsize=(2, 0.3))
    plt.gca().set_visible(False)
    cbar = plt.colorbar(sm, orientation='horizontal', fraction=0.5, pad=0.2)
    cbar.ax.tick_params(labelsize=8)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


# -------------------------
# Example usage
# -------------------------
if __name__ == '__main__':
    # user settings
    image_path = 'images/ct_synthetic.png'            # input (synthetic) image
    overlay_out = 'images/ui_mockup_overlay.png'      # overlay output
    colorbar_out = 'images/heatmap_colorbar.png'      # colorbar output
    input_size = (224, 224)                           # model input size, change if necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- load your model ---
    # Example: use a pretrained resnet for demo. Replace with your trained model and correct target layer.
    from torchvision import models
    model = models.resnet18(pretrained=True)  # <-- replace with your trained model
    model.eval()
    model.to(device)

    # choose target layer (last conv layer for ResNet18 is model.layer4[-1].conv2)
    target_layer = model.layer4[-1].conv2

    # preprocess image
    orig_pil, input_tensor, orig_size = preprocess_image(image_path, input_size=input_size)
    input_tensor = input_tensor.to(device)

    # create GradCAM object
    gradcam = GradCAM(model, target_layer)

    # compute cam (on CPU numpy)
    cam = gradcam(input_tensor, target_class=None)  # None -> uses predicted class

    # resize cam to original image size (orig_size is (W,H) from PIL)
    cam_resized = resize_cam_to_image(cam, orig_size)

    # overlay
    overlay_pil = apply_colormap_on_image(orig_pil.resize(orig_size), cam_resized, colormap=cm.jet, alpha=0.45)

    # ensure output dir exists
    os.makedirs(os.path.dirname(overlay_out), exist_ok=True)

    # save overlay
    overlay_pil.save(overlay_out, dpi=(300,300))

    # save colorbar
    save_colorbar(colormap=cm.jet, filename=colorbar_out)

    # cleanup hooks
    gradcam.close()

    print(f"Saved overlay: {overlay_out}")
    print(f"Saved colorbar: {colorbar_out}")