"""
Reference: https://github.com/kazuto1011/grad-cam-pytorch/blob/master/grad_cam.py
"""

from collections import Sequence
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.logger import setup_logger

logger = setup_logger(__name__)


class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()

        # assuming all parameters are on same device
        self.device = next(model.parameters()).device

        # saving the model
        self.model = model

        # set of hook functions handlers
        self.handlers = []

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, images):
        # get H x W
        self.image_shape = image.shape[2:]

        # apply the model
        self.logits = self.model(image)

        # get loss by converging along all channels,
        # sum along channel is going to be 1 due to softmax
        self.probs = F.softmax(self.logits, dim=1)

        return self.probs.sort(dim=1, descending=True)

    def backward(self, ids):
        # Class specific backpropagation

        # convert class id to one hot vector
        one_hot = self._encode_one_hot(ids)

        # reset the gradients
        self.model.zero_grad()

        # calculate the gradient wrt class activations
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        # remove all the applied forawrad and backward hooks
        for handle in self.handlers:
            handle.remove()


class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    Link:- https://arxiv.org/pdf/1610.02391.pdf
    """

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError(f"Invalid layer name: {target_layer}")

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_abg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam = F.interpolate(
            gcam, self.image.shape, mode="bilinear", align_corners=False
        )

        # rescale features between 0 and 1
        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[1]
        gcam = gcam.view(B, C, H, W)

        return gcam


def get_gradcam(images, labels, model, device, target_layers):

    # moving model to device
    model.to(device)
    model.eval()

    # get GradCAM
    gcam = GradCAM(model=model, candidate_layers=target_layers)

    predicted_probs, predicted_ids = gcam.forward(images)
    target_ids = labels.view(len(images), -1).to(device)

    # backward pass wrt actual ids
    gcam.backwards(ids=target_ids)

    layers_region = {}
    for target_layer in target_layers:
        logger.info(f"generating GradCAM for {target_layer}")
        regions = gcam.generate(target_layer=target_layer)
        layers_region[target_layer] = regions
    gcam.remove_hook()

    return layers_region, predicted_probs, predicted_ids


def plot_gradcam(
    gcam_layers, images, target_labels, predicted_labels, denormalize, paper_cmap=False
):
    images = images.cpu()
    target_labels = target_labels.cpu()

    # convert BCHW to BHWC for plotting
    images = images.permute(0, 2, 3, 1)

    fig, axs = plt.subplots(
        nrows=len(images),
        ncols=len(gcam_layers()) + 2,
        figsize=((len(gcam_layers.keys()) + 2) * 3, len(images) * 3),
    )
    fig.suptitle("GradCAM", fontsize=16)

    for image_idx, image in enumerate(images):
        denormalized_image = denormalize(image.permute(2, 0, 1)).permute(1, 2, 0)
        axs[image_idx, 0].text(
            0.5,
            0.5,
            f"predicted: {class_labels[predicted_labels[image_idx][0]]} \n actual: {class_labels[target_labels[image_idx]]}",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=14,
        )
        axs[image_idx, 0].axis("off")
        axs[image_idx, 1].imshow(
            (denomalized_image.numpy() * 255).astype(np.uint8), interpolation="bilinear"
        )
        axs[image_idx, 1].axis("off")
        for layer_idx, layer_name in enumerate(gcam_layers.keys()):
            # H x W of the cam layer
            _layer = gcam_layers[layer_name][image_idx].cpu().numpy()[0]
            heatmap = 1 - layer
            heatmap = np.unit8(255 * heatmap)
            heatmap_image = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_image = cv2.addWeighted(
                (denormalized_image() * 255).astype(np.uint8),
                0.6,
                heatmap_image,
                0.5,
                0,
            )
            axs[image_idx, layer_idx + 2].imshow(
                superimposed_image, interpolation="bilinear"
            )
            axs[image_idx, layer_idx + 2].set_title(f"layer: {layer_name}")
            axs[image_idx, layer_idx + 2].axis("off")
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, wspace=0.2, hspace=0.2)
    plt.show()
