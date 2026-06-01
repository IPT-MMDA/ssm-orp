import torch
import cvxpy as cp
import numpy as np
from model import VerifiableAttention


def verify_attention_softmax(model, x_clean_tensor, epsilon=0.01):
    W_q = model.W_q.weight.detach().numpy()
    W_k = model.W_k.weight.detach().numpy()
    x_clean = x_clean_tensor.detach().numpy()

    seq_len, embed_dim = x_clean.shape
    head_dim = W_q.shape[0]
    scale_factor = np.sqrt(head_dim)

    Q_clean = W_q @ x_clean[0]
    K_clean = W_k @ x_clean[1]
    pre_clean = (Q_clean.T @ K_clean) / scale_factor

    h_clean = 0.5

    grad_d0 = (W_q.T @ K_clean) / scale_factor
    grad_d1 = (W_k.T @ Q_clean) / scale_factor

    norm_Wq = np.linalg.norm(W_q, ord=2)
    norm_Wk = np.linalg.norm(W_k, ord=2)
    max_quad_error = (norm_Wq * norm_Wk * (epsilon**2)) / scale_factor

    n_vars = 1 + embed_dim + embed_dim + 1
    X = cp.Variable((n_vars, n_vars), symmetric=True)

    idx_const = 0
    idx_d0_start, idx_d0_end = 1, 1 + embed_dim
    idx_d1_start, idx_d1_end = 1 + embed_dim, 1 + 2 * embed_dim
    idx_h = 1 + 2 * embed_dim

    constraints = [X >> 0, X[idx_const, idx_const] == 1.0]

    trace_d0 = cp.trace(X[idx_d0_start:idx_d0_end, idx_d0_start:idx_d0_end])
    trace_d1 = cp.trace(X[idx_d1_start:idx_d1_end, idx_d1_start:idx_d1_end])
    constraints.append(trace_d0 <= epsilon**2)
    constraints.append(trace_d1 <= epsilon**2)

    pre_expr_upper = pre_clean + max_quad_error
    for i in range(embed_dim):
        pre_expr_upper += grad_d0[i] * X[idx_const, idx_d0_start + i]
        pre_expr_upper += grad_d1[i] * X[idx_const, idx_d1_start + i]

    constraints.append(X[idx_const, idx_h] - h_clean <= 0.25 * (pre_expr_upper - pre_clean))

    constraints.append(X[idx_const, idx_h] >= 0.0)
    constraints.append(X[idx_const, idx_h] <= 1.0)

    objective = cp.Maximize(X[idx_const, idx_h])
    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.SCS, verbose=False)
        if problem.status in ["optimal", "optimal_inaccurate"]:
            return float(problem.value), problem.status
        else:
            return None, problem.status
    except Exception as e:
        return None, str(e)


if __name__ == "__main__":
    torch.manual_seed(1)

    SEQ_LEN = 2
    EMBED_DIM = 4
    HEAD_DIM = 4

    model = VerifiableAttention(embed_dim=EMBED_DIM, head_dim=HEAD_DIM)
    x_clean = torch.randn(SEQ_LEN, EMBED_DIM)

    verify_attention_softmax(model, x_clean, epsilon=0.01)
    verify_attention_softmax(model, x_clean, epsilon=0.1)
    verify_attention_softmax(model, x_clean, epsilon=1.5)
