# IQC-Inspired Robustness Verification for Transformer Self-Attention

**Reference Paper:** [Safety Verification and Robustness Analysis of Neural Networks via Quadratic Constraints and Semidefinite Programming](https://arxiv.org/abs/1903.01287)
**Authors:** Mahyar Fazlyab, Manfred Morari, George J. Pappas

## Abstract
This experiment adapts continuous robustness analysis to the discrete, sequential domain of NLP. Specifically, it verifies the robustness of the Transformer Self-Attention mechanism under bounded input perturbations (such as typos or synonym replacements mapped to $L_2$ embedding spheres).

## The Challenge: The Quadratic Bottleneck
Certifying RNNs typically involves bounding linear pre-activations. However, the Self-Attention mechanism computes scores quadratically:
$Pre = (W_q X)^T (W_k X)$

Passing perturbed inputs $(X + \delta)$ through this operation creates multiplied uncertainty terms $(\delta_i \delta_j)$. Standard SDP moment matrices cannot resolve these cubic moments, leading to overly loose bounds (e.g., trivially returning 1.0).

## The Solution: Linearization with Guaranteed Error Bounds
To bypass the cubic-moment limitation while maintaining strict mathematical guarantees, this implementation utilizes **Gradient-Based Linearization**:
1. Computes the exact linear shift of the pre-activation using the gradients of the $Q$ and $K$ projection matrices.
2. Calculates the absolute worst-case quadratic error bound mathematically: $||W_q|| \cdot ||W_k|| \cdot \epsilon^2$.
3. Applies the Softmax Lipschitz constraints described in Proposition 2(f) of the DeepSDP framework to the linearized upper bound.

This allows the CVXPY solver to find strict, non-trivial maximum bounds for attention weights under adversarial embedding perturbations.

## Results
The evaluation simulates embedding space shifts ranging from minor typos to massive adversarial context swaps.

| Scenario | Epsilon ($L_2$ shift) | Certified Max Attention | Solver Status |
| :--- | :--- | :--- | :--- |
| Clean (No Attack) | 0.0001 | 0.5002 | optimal_inaccurate |
| Single Char Typo | 0.0500 | 0.5083 | optimal |
| Synonym (Close) | 0.2000 | 0.5345 | optimal |
| Synonym (Distant) | 0.5000 | 0.5922 | optimal |
| Adversarial Swap | 1.2000 | 0.7546 | optimal |
| Complete Context Shift | 2.5000 | 1.0000 | optimal |

As demonstrated, the SDP verifier successfully proves that minor input perturbations ($\epsilon \le 0.2$) cannot drastically alter the attention distribution, providing a strict mathematical certificate of safety for the attention head.

## Reproducibility
* `model.py`: Verifiable PyTorch implementation of the Self-Attention layer.
* `sdp_verifier.py`: The SDP formulation using `CVXPY` and Softmax Quadratic Constraints.
* `evaluate_attention.py`: Pipeline connecting NLP attack scenarios to continuous $\epsilon$-radiuses.
