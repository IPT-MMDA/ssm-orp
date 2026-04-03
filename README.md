# Robustness of State Space Models via Lipschitz Regularization
## 1. Executive Summary
This repository explores the stability of State Space Models (SSMs) under noisy input conditions. Standard Recurrent Neural Networks and SSMs often suffer from "explosive" hidden states when internal transition matrices have eigenvalues greater than 1. By enforcing a 1-Lipschitz bound on the transition matrix A using Spectral Normalization, we demonstrate a significant increase in noise resilience without sacrificing the model's ability to learn long-range dependencies.

## 2. Problem Statement: The Stability Gap
In a discrete-time SSM, the hidden state update is defined as:

$$h_t = \sigma(A h_{t-1} + B x_t)$$

If the spectral norm of A (its largest singular value) exceeds 1, small perturbations in the input $$x_t$$ can be amplified exponentially over time. In real-world applications (e.g., signal processing or medical telemetry), this leads to high variance and model failure.

## 3. Methodology
### 3.1 Dataset: The Adding Task

To test the "memory" and stability of the models, we utilized the Adding Task. The model must track two specific values in a long sequence (L=50) and return their sum. This requires the hidden state to remain stable over many timesteps.

### 3.2 Regularization Technique

We applied Spectral Normalization to the following parameters:

Transition Matrix A: Enforcing σ(A)≤1.0 to ensure BIBO (Bounded-Input, Bounded-Output) stability.

Input Matrix B and Output Matrix C: Normalized to ensure the entire mapping remains Lipschitz-continuous.

## 4. Experimental Results
We evaluated the models by injecting Gaussian white noise into the input sequences and measuring the Mean Euclidean Deviation (Δh) of the hidden states.

| Model | Hidden Δh | Standard Deviation (σ)| Stability |
| :--- | :---: | ---: | ---: |
| Baseline | 2.58 | ± 0.1530 | Low |
| Regularized | 0.79 | ± 0.0689 | High |

### Stability Analysis

As shown in the plot below, the Baseline model exhibits a near-linear sensitivity to noise. In contrast, the Regularized model maintains a bounded response, confirming that the Lipschitz constraint effectively "shrouds" the hidden state from external perturbations.
![Stability Plot](./stability_plot.png)

## 5. Implementation & Verification
### Framework: PyTorch

Verification: pytest was used to confirm that the spectral norm of the transition matrices remained within the theoretical bound (≤1.05) after training.

### Reproducibility:

Bash
python src/train.py    # Trains and saves models
python src/evaluate.py # Generates statistics and stability_plot.png
pytest tests/          # Validates mathematical constraints
## 6. Conclusion
The results confirm that Spectral Normalization is a potent tool for stabilizing SSMs. While a minor trade-off in training expressivity was observed (slightly higher initial loss), the resulting model is significantly more reliable for deployment in high-noise environments.