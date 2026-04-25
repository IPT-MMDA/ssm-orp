from .solver import solve_sdp

def certify(model, W, b, x, y, eps):
    return solve_sdp(W, b, x, y, eps)