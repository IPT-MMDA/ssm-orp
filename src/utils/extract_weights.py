def extract_weights(model):
    W, b = [], []

    for layer in model.layers:
        if hasattr(layer, "weight"):
            W.append(layer.weight.detach().cpu().numpy())
            b.append(layer.bias.detach().cpu().numpy())

    return W, b