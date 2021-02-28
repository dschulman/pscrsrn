import torch.optim.lr_scheduler as olrs

def WarmupExponentialLR(optimizer, gamma, warmup_epochs, warmup_start=0.01, last_epoch=-1, verbose=False):
    def _lr_lambda(epoch):
        if epoch < warmup_epochs:
            return warmup_start + ((1.0 - warmup_start) * epoch / warmup_epochs)
        else:
            return gamma ** (epoch - warmup_epochs)
    return olrs.LambdaLR(optimizer, _lr_lambda, last_epoch, verbose)

def WarmupStepLR(optimizer, step_size, gamma, warmup_epochs, warmup_start=0.01, last_epoch=-1, verbose=False):
    def _lr_lambda(epoch):
        if epoch < warmup_epochs:
            return warmup_start + ((1.0 - warmup_start) * epoch / warmup_epochs)
        else:
            return gamma ** ((epoch - warmup_epochs) // step_size)
    return olrs.LambdaLR(optimizer, _lr_lambda, last_epoch, verbose)
