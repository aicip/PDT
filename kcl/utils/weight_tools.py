import numpy as np
import torch


def flat_params_as_numpy(network):
    weights = []
    for param in network.parameters():
        weights.append(param.view(-1))
    return torch.cat(weights, 0).cpu().detach().numpy()


def flat_params_as_torch(network):
    weights = []
    for param in network.parameters():
        weights.append(param.view(-1))
    return torch.cat(weights, 0).detach()


@torch.no_grad()
def flat_param_groups_as_torch(param_groups):
    weights = []
    for param_group in param_groups:
        for param in param_group['params']:
            weights.append(param.view(-1))
    return torch.cat(weights, 0).detach()


@torch.no_grad()
def add_diff_to_param_groups(param_groups, diff):
    i = 0
    for param_group in param_groups:
        for param in param_group['params']:
            param.data.add_(diff[i:i + param.numel()].view(param.size()))
            i += param.numel()
    assert i == diff.numel(), "diff size mismatch"


def flat_grads_as_torch(network):
    grads = []
    for param in network.parameters():
        grads.append(param.grad.view(-1))
    return torch.cat(grads, 0).detach()


def array_to_params(network, params, device='cpu', filter_grad=False):
    if isinstance(params, np.ndarray):
        params = torch.from_numpy(params).to(device)
    i = 0
    with torch.no_grad():
        for weight in network.parameters():
            if filter_grad and not weight.requires_grad:
                continue
            weight.data.copy_(params[i:i + weight.numel()].view(weight.size()))
            i += weight.numel()
    assert i == params.numel(), "params size mismatch"
