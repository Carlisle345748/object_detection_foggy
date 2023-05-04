import logging

from detectron2.engine import HookBase
from detectron2.utils.events import get_event_storage
import torch.nn as nn


class GradMonitor(HookBase):
    def __init__(self, model: nn.Module):
        self.model = model

    def after_backward(self):
        logger = logging.getLogger(__name__)
        logger.parent = logging.getLogger('detectron2')
        gradient_norms = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradient_norms[name] = param.grad.data.norm(2).item()

        exploding_threshold = 1e+5
        vanishing_threshold = 1e-5

        min_grad_norm, max_grad_norm = None, None
        for name, grad_norm in gradient_norms.items():
            if min_grad_norm is None or grad_norm < min_grad_norm:
                min_grad_norm = grad_norm
            if max_grad_norm is None or grad_norm > max_grad_norm:
                max_grad_norm = grad_norm

            if grad_norm > exploding_threshold:
                logger.info(f"Gradient exploding detected in layer {name}: {grad_norm}")
            elif grad_norm < vanishing_threshold:
                logger.info(f"Gradient vanishing detected in layer {name}: {grad_norm}")

        storage = get_event_storage()
        storage.put_scalar("min_grad_norm", min_grad_norm)
        storage.put_scalar("max_grad_norm", max_grad_norm)
