import argparse
import random

import numpy as np
import torch


def pretty_size(num_bytes):
    """Format a number of bytes as a human-readable string with binary prefixes."""
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    value = float(num_bytes)
    idx = 0
    while value >= 1024.0 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(value)} {units[idx]}"
    return f"{value:.1f} {units[idx]}"


def full_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def unpack_batch(batch, device=None):
    """Return ``(data, labels)`` moved to ``device`` for a ``(data, labels)`` batch."""
    if len(batch) == 2:
        return batch[0].to(device), batch[1].to(device)
    raise RuntimeError("Unexpected batch structure with {} elements".format(len(batch)))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")
