import torch


def to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    elif isinstance(x, list) or isinstance(x, tuple):
        x_type = type(x)
        res = [to_device(item, device) for item in x]
        return x_type(res)
    elif isinstance(x, dict):
        res = {k: to_device(v, device) for k, v in x.items()}
        return res
    return x
