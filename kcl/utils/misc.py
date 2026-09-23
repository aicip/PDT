import random
import subprocess
import warnings

import numpy as np
import torch
import argparse


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
    # torch.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def unpack_batch(batch, device=None):
    if len(batch) == 1:
        data = batch[0]['data']
        labels = batch[0]['label'].reshape((-1, )).long()
    elif len(batch) == 2:
        data = batch[0].to(device)
        labels = batch[1].to(device)
    else:
        raise RuntimeError("Invalid batch size: {}".format(len(batch)))

    return data, labels


def query_gpu_memory():
    try:
        # Run the nvidia-smi command and capture its output
        output = subprocess.check_output(
            [
                'nvidia-smi',
                '--query-gpu=memory.total,memory.used,memory.free',
                '--format=csv,nounits,noheader'
            ],
            universal_newlines=True
        )

        # Split the output into lines
        lines = output.strip().split('\n')

        # Parse the values from the last line and convert to integers
        values = list(map(int, lines[-1].split(',')))

        # Return the values as a tuple (TotalMemory, UsedMemory, FreeMemory)
        return tuple(values)

    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")
        return None

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')