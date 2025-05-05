# interval_propagation.py
import torch
import torch.nn as nn

def relu_bounds(l, u):
    return torch.clamp(l, min=0), torch.clamp(u, min=0)

def linear_bounds(W, b, l_in, u_in):
    W_pos = torch.clamp(W, min=0)
    W_neg = torch.clamp(W, max=0)
    l_out = W_pos @ l_in + W_neg @ u_in + b
    u_out = W_pos @ u_in + W_neg @ l_in + b
    return l_out, u_out

def propagate_bounds(model, x0, epsilon):
    l = x0 - epsilon
    u = x0 + epsilon
    l = l.view(-1)
    u = u.view(-1)

    for layer in model.net:
        if isinstance(layer, nn.Linear):
            W = layer.weight.detach()
            b = layer.bias.detach()
            l, u = linear_bounds(W, b, l, u)
        elif isinstance(layer, nn.ReLU):
            l, u = relu_bounds(l, u)
        elif isinstance(layer, nn.Flatten):
            continue
        else:
            raise NotImplementedError(f"Layer {layer} not supported")
    return l, u  # Bounds on logits
