import torch


def load_model(model_class, checkpoint_path=None, model_args=None, device=None):
    init_args = model_class.__init__.__code__.co_varnames[1:model_class.__init__.__code__.co_argcount]

    if model_args is None:
        if checkpoint_path is None:
            raise ValueError('Either checkpoint or model_args must be provided')

        checkpoint = torch.load(checkpoint_path)
        model_args = checkpoint.get('model_args')
        if model_args is None:
            raise ValueError('model_args not found in checkpoint')

    model_args = {k: v for k, v in model_args.items() if k in init_args}
    model = model_class(**model_args)
    if device is not None:
        model.to(device)
        model.device = device
    return model


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def load_optimizer(optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return optimizer
