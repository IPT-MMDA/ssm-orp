import cvxpy as cp
import numpy as np

# def solve_sdp(W, b, x0, y_true, eps, solver=cp.CLARABEL):

#     n0, n1, nf = W[0].shape[1], W[0].shape[0], W[1].shape[0]

#     x0 = np.asarray(x0).reshape(-1)
#     x_min, x_max = x0 - eps, x0 + eps

#     gamma = cp.Variable(n0, nonneg=True)

#     P = cp.bmat([
#         [cp.diag(-2 * gamma),
#          cp.reshape(cp.multiply(gamma, x_min + x_max), (n0, 1))],
#         [cp.reshape(cp.multiply(gamma, x_min + x_max), (1, n0)),
#          cp.reshape(-2 * cp.sum(cp.multiply(gamma, x_min * x_max)), (1, 1))]
#     ])

#     E0 = np.block([
#         [np.eye(n0), np.zeros((n0, n1)), np.zeros((n0, 1))],
#         [np.zeros((1, n0)), np.zeros((1, n1)), np.ones((1, 1))]
#     ])

#     M_in = E0.T @ P @ E0

#     lam = cp.Variable(n1, nonneg=True)
#     nu = cp.Variable(n1, nonneg=True)
#     eta = cp.Variable(n1, nonneg=True)
#     T = cp.Variable((n1, n1), symmetric=True)

#     LamT = cp.diag(lam) + T

#     Q = cp.bmat([
#         [np.zeros((n1, n1)), LamT, cp.reshape(-nu, (n1, 1))],
#         [LamT.T, -2 * LamT, cp.reshape(nu + eta, (n1, 1))],
#         [cp.reshape(-nu, (1, n1)), cp.reshape(nu + eta, (1, n1)), np.zeros((1, 1))]
#     ])

#     C = np.block([
#         [W[0], np.zeros((n1, n1)), b[0].reshape(-1, 1)],
#         [np.zeros((n1, n0)), np.eye(n1), np.zeros((n1, 1))],
#         [np.zeros((1, n0)), np.zeros((1, n1)), np.ones((1, 1))]
#     ])

#     M_mid = C.T @ Q @ C

#     constraints_base = [
#         T <= 0,
#         cp.diag(T) == 0,
#         cp.norm(T, "fro") <= 10
#     ]

#     for i in range(nf):
#         if i == y_true:
#             continue

#         c = np.zeros((nf, 1))
#         c[i, 0] = 1
#         c[y_true, 0] = -1

#         S = np.block([
#             [np.zeros((n0, n0 + nf)), np.zeros((n0, 1))],
#             [np.zeros((nf, n0 + nf)), c],
#             [np.zeros((1, n0)), c.T, np.zeros((1, 1))]
#         ])

#         D = np.block([
#             [np.eye(n0), np.zeros((n0, n1)), np.zeros((n0, 1))],
#             [np.zeros((nf, n0)), W[1], b[1].reshape(-1, 1)],
#             [np.zeros((1, n0)), np.zeros((1, n1)), np.ones((1, 1))]
#         ])

#         LMI = M_in + M_mid + (D.T @ S @ D)

#         prob = cp.Problem(
#             cp.Minimize(0),
#             constraints_base + [LMI << -1e-10 * np.eye(LMI.shape[0])]
#         )

#         try:
#             prob.solve(solver=solver)
#             if prob.status not in ["optimal", "optimal_inaccurate"]:
#                 return 0
#         except:
#             return 0

#     return 1


def solve_sdp(W, b, x0, y_true, eps, solver=cp.CLARABEL):

    n0, n1, nf = W[0].shape[1], W[0].shape[0], W[1].shape[0]

    x0 = np.asarray(x0).reshape(-1)
    x_min, x_max = x0 - eps, x0 + eps

    gamma = cp.Variable(n0, nonneg=True)

    P = cp.bmat([
        [cp.diag(-2 * gamma),
         cp.reshape(cp.multiply(gamma, x_min + x_max), (n0, 1))],
        [cp.reshape(cp.multiply(gamma, x_min + x_max), (1, n0)),
         cp.reshape(-2 * cp.sum(cp.multiply(gamma, x_min * x_max)), (1, 1))]
    ])

    E0 = np.block([
        [np.eye(n0), np.zeros((n0, n1)), np.zeros((n0, 1))],
        [np.zeros((1, n0)), np.zeros((1, n1)), np.ones((1, 1))]
    ])

    M_in = E0.T @ P @ E0

    lam = cp.Variable(n1, nonneg=True)
    nu = cp.Variable(n1, nonneg=True)
    eta = cp.Variable(n1, nonneg=True)
    T = cp.Variable((n1, n1), symmetric=True)

    LamT = cp.diag(lam) + T

    Q = cp.bmat([
        [np.zeros((n1, n1)), LamT, cp.reshape(-nu, (n1, 1))],
        [LamT.T, -2 * LamT, cp.reshape(nu + eta, (n1, 1))],
        [cp.reshape(-nu, (1, n1)), cp.reshape(nu + eta, (1, n1)), np.zeros((1, 1))]
    ])

    C = np.block([
        [W[0], np.zeros((n1, n1)), b[0].reshape(-1, 1)],
        [np.zeros((n1, n0)), np.eye(n1), np.zeros((n1, 1))],
        [np.zeros((1, n0)), np.zeros((1, n1)), np.ones((1, 1))]
    ])

    M_mid = C.T @ Q @ C

    constraints_base = [
        T <= 0,
        cp.diag(T) == 0,
        cp.norm(T, "fro") <= 10
    ]

    logits = W[1] @ np.maximum(W[0] @ x0 + b[0], 0) + b[1]
    top_k = np.argsort(logits)[-3:]
    candidates = [i for i in top_k if i != y_true][:2]
    
    if len(candidates) == 0:
        return 1

    for i in candidates:

        c = np.zeros((nf, 1))
        c[i, 0] = 1
        c[y_true, 0] = -1

        S = np.block([
            [np.zeros((n0, n0 + nf)), np.zeros((n0, 1))],
            [np.zeros((nf, n0 + nf)), c],
            [np.zeros((1, n0)), c.T, np.zeros((1, 1))]
        ])

        D = np.block([
            [np.eye(n0), np.zeros((n0, n1)), np.zeros((n0, 1))],
            [np.zeros((nf, n0)), W[1], b[1].reshape(-1, 1)],
            [np.zeros((1, n0)), np.zeros((1, n1)), np.ones((1, 1))]
        ])

        LMI = M_in + M_mid + (D.T @ S @ D)

        prob = cp.Problem(
            cp.Minimize(0),
            constraints_base + [LMI << -1e-10 * np.eye(LMI.shape[0])]
        )

        try:
            prob.solve(solver=solver, max_iter=1000)
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                return 0
        except:
            return 0

    return 1