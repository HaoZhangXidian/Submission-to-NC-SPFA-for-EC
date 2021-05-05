from torch import optim


def build_optimizers(model, config):
    optimizer = config['training']['optimizer']
    lr = config['training']['lr']

    params = model.parameters()

    # Optimizers
    if optimizer == 'rmsprop':
        optimizer = optim.RMSprop(params, lr=lr, alpha=0.99, eps=1e-8)
    elif optimizer == 'adam':
        optimizer = optim.Adam(params, lr=lr)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=lr, momentum=0.)

    return optimizer
