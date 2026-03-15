import torch
import torch.nn as nn
import numpy as np
import random
import os

# Step 1: Utility helpers for reproducibility and regularization.
# These helpers are imported and used by the training script.

def seed_everything(seed=42):
    """Set global random seeds for reproducible experiments.

    This function seeds Python, NumPy, and PyTorch randomness.

    Important: True determinism depends on the hardware/driver and some
    operations may remain non-deterministic.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    print(f"Global seed set to {seed}")

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Apply DropPath (stochastic depth) regularization.

    Args:
        x: Input tensor.
        drop_prob: Probability of dropping paths.
        training: Whether in training mode.
        scale_by_keep: Whether to scale keeping paths to preserve expected sum.

    Returns:
        Tensor with dropped paths (or the original tensor in eval mode).

    Important: DropPath should only be enabled during training.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """nn.Module wrapper for DropPath (stochastic depth).

    This class is used in model blocks to apply stochastic depth as a layer.
    It delegates to `drop_path` and respects the module's training flag.
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        """Apply DropPath regularization depending on module `training` state."""
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
