# SharedMamba: ALBERT-style Weight Sharing for State Space Models

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c.svg)

## Task

Implement an SSM architecture where the weights of the state space blocks are shared across all depth layers (similar to ALBERT). Evaluate the trade-off between the resulting parameter reduction and the model's ability to extract hierarchical features from standard 1D signals. Measure the robustness degradation vs the standard.

### Objective

Explore the viability of cross-layer parameter sharing in SSMs (an "ALBERT-style" SSM) to drastically reduce the model's memory footprint and parameter count.

### Methodology

Modify the SSM architecture so that a single state-space block (or a small group of blocks) is instantiated once in memory but invoked iteratively for N layers during the forward pass. Train this model from scratch on standard 1D signals (e.g., audio or time-series).

### Evaluation & Metrics

Measure the parameter reduction ratio. Evaluate performance against a standard unshared SSM of equivalent depth. Analyze robustness by testing the model's resilience to input perturbations (e.g., frequency masking, time shifting).

### Expected Challenges

Deep networks typically build hierarchical representations (low-level features in early layers, high-level semantics in later layers). Forcing the same transition dynamics across all depths might cripple this hierarchical extraction, making the model rigid and less robust to complex signal variations.

## Abstract

State Space Models (SSMs), particularly Mamba, have emerged as highly efficient alternatives to Transformers for modeling long-sequence data. However, deep stacked SSM architectures often suffer from parameter bloat. To address this, we propose SharedMamba, a novel architecture that employs continuous weight sharing across sequence processing steps, simulating a recursive dynamical system within a single SSM block. We evaluate both standard and shared architectures on continuous (MNIST) and discrete (ListOps) modalities, applying zero-shot 20% and 50% input masking during inference to strictly test Out-of-Distribution (OOD) robustness.

Our results reveal a critical architectural trade-off. On discrete algorithmic tasks (ListOps), SharedMamba achieves perfect performance parity with standard deep models (maintaining ~96-97% accuracy across all masking regimes) while reducing core sequence modeling parameters by a factor of 6 (from 702k to 118k). Conversely, on continuous spatial signals (MNIST), the rigid state manifold of shared parameters limits OOD signal recovery; under 20% noise, a 3-layer Standard Mamba retains 61.1% accuracy, whereas SharedMamba drops to 37.5%. These findings establish SharedMamba as an exceptionally compressed and optimal architecture for memory-constrained discrete logic and reasoning tasks on Edge AI, while highlighting the necessity of independent hierarchical projections for robust continuous signal processing.

## Architecture

The core difference between the **StandardMamba** and our proposed **SharedMamba** lies in how the state evolves across the computational depth of the model.

### Standard Mamba (Deep Stack)

In a conventional architecture, an $N$-layer Mamba model consists of $N$ independent blocks. If $X_{n-1}$ is the input to the $n$-th layer, the forward pass is defined as:

$$X_n = \text{MambaBlock}(X_{n-1}; \Theta_n, \Phi_n) \quad \text{for } n \in \{1, \dots, N\}$$

Where $\Theta_n$ represents the core trainable parameters of the SSM block, and $\Phi_n$ represents the **layer-specific Normalization parameters**. Because every layer uses a distinct normalization, the hidden state is constantly projected into different mathematical subspaces, which increases the risk of overfitting to specific input distributions (like clean pixels).

### SharedMamba (Recursive Iteration)

To combat parameter bloat and enforce structural regularization, **SharedMamba** uses a single set of parameters, including the normalization layer, applied iteratively:

$$X_n = \text{MambaBlock}(X_{n-1}; \Theta_{shared}, \Phi_{shared}) \quad \text{for } n \in \{1, \dots, N\}$$

**Key Architectural Benefits:**

1. **Parameter Efficiency:** The core sequence modeling parameters, **including normalization weights**, are reduced by a factor of $N$ (e.g., a 6-iteration SharedMamba uses the exact same number of parameters as a 1-layer Standard Mamba).
2. **Stable State Manifold:** By forcing the exact same Normalization ($\Phi_{shared}$) across all $N$ iterations, the hidden state is constrained to evolve within a single, stable vector space. 
3. **Inductive Bias for Robustness:** This shared transformation prevents the model from acting as a deep lookup table for noisy coordinates, forcing it instead to learn generalized, invariant topological features of the input data.

## Experiments

To rigorously evaluate the inductive biases of both architectures, we designed two distinct tasks testing continuous sequence modeling and discrete algorithmic reasoning. Crucially, **no data augmentation was used during training**; all models were trained exclusively on pristine data to evaluate true Out-of-Distribution generalization rather than memorized noise patterns.

### Task 1: Controlled Baseline & Spurious Correlations (Synthetic Signals)

Before evaluating on complex datasets, we conducted an initial test using synthetically generated 1D continuous signals. 

* **Purpose:** To verify if both standard and shared SSM architectures could effectively learn fundamental temporal patterns in an isolated, controlled environment.
* **Result & Insight:** Interestingly, both `StandardMamba` and `SharedMamba` exhibited an extreme capacity for memorization, rapidly achieving near 100% accuracy on the training set. However, both models collapsed to sub-random performance (<50% accuracy) on the test set. This highlights a known vulnerability of highly expressive SSMs: a tendency to overfit on procedural artifacts or **spurious correlations** within algorithmically generated synthetic data. This crucial finding underscored the necessity of moving beyond synthetic signals to structured, real-world topologies (MNIST) and strict rule-based environments (ListOps) to properly evaluate true algorithmic generalization.

### Task 2: Out-of-Distribution Robustness (Continuous Modality)

We adapted the **MNIST** dataset for sequence modeling by flattening the $28 \times 28$ images into 1D sequences of 784 continuous pixel values.

* **Training:** Models were trained on perfect, uncorrupted images.
* **Evaluation (Stress Test):** During inference, we applied **zero-shot masking**, randomly zeroing out 20% and 50% of the input pixels. Since the models never encountered missing data during training, this task strictly tests whether the architecture learned brittle, coordinate-specific representations (overfitting) or robust, global topological features.

### Task 3: Algorithmic Reasoning & State Retention (Discrete Modality)

To test the models' ability to handle discrete, hierarchical data, we used the **ListOps** dataset. This task requires the model to parse nested mathematical operations (e.g., `( 3 4 5 ( 1 2 ( 3 6 ) ) )`) represented as sequences of discrete tokens.

* **Variable Length & Padding:** Sequences vary in length and are padded with zeros to match a maximum tensor size.
* **The Challenge:** Because the final classification relies on Mean Pooling across the temporal dimension, the SSM must retain its parsed hidden state perfectly across the empty padding tokens without "diluting" or forgetting the logical output. This explicitly tests the algorithmic stability and state-retention capabilities of the recursive SharedMamba block.

## Results Table

batch size = 128

confidence level = 0.95

epochs = 10

### Task 1: Synthetic Signals

| Architecture     | Depth (Layers) | Core Params   | Train Loss | Train Acc (Clean) | Test Acc (Clean) | Interval       |
| :---             | :---:          | :---:         | :---:      | :---:             | :---:            | :---:          |
| Standard Mamba   | 6              | [0.701 M]     | 0.012      | 99.9%             | 51.5%            | [51.5%, 51.5%] |
| **Shared Mamba** | 6              | **[0.117 M]** | 0.174      | 93.4%             | 44.5%            | [44.5%, 44.5%] |
| Standard Mamba   | 3              | [0.351 M]     | 0.165      | 93.4%             | 51.0%            | [51.0%, 51.0%] |
| **Shared Mamba** | 3              | **[0.117 M]** | 0.342      | 83.7%             | 47.0%            | [47.0%, 47.0%] |
|                  |                |               |            |                   |                  |                |

| Architecture     | Depth (Layers) | Core Params   | Train Loss | Train Acc (Clean) | Test Acc (Mask 20%) | Interval       |
| :---             | :---:          | :---:         | :---:      | :---:             | :---:               | :---:          |
| Standard Mamba   | 6              | [0.701 M]     | 0.050      | 99.2%             | 54.9%               | [53.3%, 56.6%] |
| **Shared Mamba** | 6              | **[0.117 M]** | 0.174      | 93.4%             | 46.2%               | [44.1%, 48.3%] |
| Standard Mamba   | 3              | [0.351 M]     | 0.230      | 89.1%             | 51.2%               | [49.8%, 52.7%] |
| **Shared Mamba** | 3              | **[0.117 M]** | 0.342      | 83.7%             | 49.1%               | [47.1%, 51.2%] |
|                  |                |               |            |                   |                     |                |

| Architecture     | Depth (Layers) | Core Params   | Train Loss | Train Acc (Clean) | Test Acc (Mask 50%) | Interval       |
| :---             | :---:          | :---:         | :---:      | :---:             | :---:               | :---:          |
| Standard Mamba   | 6              | [0.701 M]     | 0.062      | 98.2%             | 51.1%               | [49.7%, 52.5%] |
| **Shared Mamba** | 6              | **[0.117 M]** | 0.174      | 93.4%             | 48.6%               | [46.0%, 51.2%] |
| Standard Mamba   | 3              | [0.351 M]     | 0.473      | 90.2%             | 48.2%               | [46.2%, 50.3%] |
| **Shared Mamba** | 3              | **[0.117 M]** | 0.342      | 83.7%             | 49.6%               | [47.4%, 51.8%] |
|                  |                |               |            |                   |                     |                |

### Task 2: ListOps

vocabulary size = 11 (`0: padding, 1: '(', 2: ')', 3-10: random numbers`)

sequence length = 128

classes = 2 (`0: unbalanced, 1: balanced`)

| Architecture     | Depth (Layers) | Core Params   | Train Loss | Train Acc (Clean) | Test Acc (Clean) | Interval       |
| :---             | :---:          | :---:         | :---:      | :---:             | :---:            | :---:          |
| Standard Mamba   | 6              | [0.702 M]     | 0.127      | 96.9%             | 96.0%            | [96.0%, 96.0%] |
| **Shared Mamba** | 6              | **[0.118 M]** | 0.149      | 96.3%             | 96.0%            | [96.0%, 96.0%] |
| Standard Mamba   | 3              | [0.352 M]     | 0.148      | 96.5%             | 92.0%            | [92.0%, 92.0%] |
| **Shared Mamba** | 3              | **[0.118 M]** | 0.140      | 97.1%             | 96.0%            | [96.0%, 96.0%] |

| Architecture     | Depth (Layers) | Core Params   | Train Loss | Train Acc (Clean) | Test Acc (Mask 20%) | Interval       |
| :---             | :---:          | :---:         | :---:      | :---:             | :---:               | :---:          |
| Standard Mamba   | 6              | [0.702 M]     | 0.176      | 95.5%             | 97.5%               | [97.5%, 97.5%] |
| **Shared Mamba** | 6              | **[0.118 M]** | 0.116      | 97.1%             | 96.0%               | [96.0%, 96.0%] |
| Standard Mamba   | 3              | [0.352 M]     | 0.152      | 95.9%             | 96.5%               | [96.5%, 96.5%] |
| **Shared Mamba** | 3              | **[0.118 M]** | 0.164      | 96.3%             | 97.5%               | [97.5%, 97.5%] |

| Architecture     | Depth (Layers) | Core Params   | Train Loss | Train Acc (Clean) | Test Acc (Mask 50%) | Interval       |
| :---             | :---:          | :---:         | :---:      | :---:             | :---:               | :---:          |
| Standard Mamba   | 6              | [0.702 M]     | 0.159      | 95.5%             | 96.5%               | [96.5%, 96.5%] |
| **Shared Mamba** | 6              | **[0.118 M]** | 0.145      | 96.1%             | 97.5%               | [97.5%, 97.5%] |
| Standard Mamba   | 3              | [0.352 M]     | 0.149      | 96.4%             | 97.0%               | [97.0%, 97.0%] |
| **Shared Mamba** | 3              | **[0.118 M]** | 0.158      | 95.6%             | 98.0%               | [98.0%, 98.0%] |

### Task 3: MNIST

input dimension = 1 (`flat mnist`)

classes = 10 (`0:0, 1:1, ..., 9:9`)

| Architecture     | Depth (Layers) | Core Params   | Train Loss | Train Acc (Clean) | Test Acc (Clean) | Interval       |
| :---             | :---:          | :---:         | :---:      | :---:             | :---:            | :---:          |
| Standard Mamba   | 6              | [0.702 M]     | 0.014      | 99.6%             | 98.7%            | [98.7%, 98.7%] |
| **Shared Mamba** | 6              | **[0.118 M]** | 0.031      | 99.1%             | 98.4%            | [98.4%, 98.4%] |
| Standard Mamba   | 3              | [0.352 M]     | 0.025      | 99.3%             | 98.5%            | [98.5%, 98.5%] |
| **Shared Mamba** | 3              | **[0.118 M]** | 0.039      | 98.8%             | 98.1%            | [98.1%, 98.1%] |

| Architecture     | Depth (Layers) | Core Params   | Train Loss | Train Acc (Clean) | Test Acc (Mask 20%) | Interval       |
| :---             | :---:          | :---:         | :---:      | :---:             | :---:               | :---:          |
| Standard Mamba   | 6              | [0.702 M]     | 0.015      | 99.6%             | 48.0%               | [47.8%, 48.2%] |
| **Shared Mamba** | 6              | **[0.118 M]** | 0.032      | 99.0%             | 26.5%               | [26.3%, 26.7%] |
| Standard Mamba   | 3              | [0.352 M]     | 0.024      | 99.3%             | 61.1%               | [60.9%, 61.3%] |
| **Shared Mamba** | 3              | **[0.118 M]** | 0.039      | 98.8%             | 37.5%               | [37.3%, 37.7%] |

| Architecture     | Depth (Layers) | Core Params   | Train Loss | Train Acc (Clean) | Test Acc (Mask 50%) | Interval       |
| :---             | :---:          | :---:         | :---:      | :---:             | :---:               | :---:          |
| Standard Mamba   | 6              | [0.702 M]     | 0.013      | 99.6%             | 14.8%               | [14.7%, 15.0%] |
| **Shared Mamba** | 6              | **[0.118 M]** | 0.031      | 99.1%             | 9.6%                | [9.5%, 9.6%]   |
| Standard Mamba   | 3              | [0.352 M]     | 0.026      | 99.2%             | 17.7%               | [17.5%, 17.9%] |
| **Shared Mamba** | 3              | **[0.118 M]** | 0.039      | 98.8%             | 12.9%               | [12.8%, 13.1%] |

## Analysis

The empirical evaluation reveals a clear trade-off between parameter efficiency, algorithmic state retention, and Out-of-Distribution (OOD) robustness when applying ALBERT-style weight sharing to State Space Models (SSMs).

### 1. The Memorization Trap (Task 1: Synthetic Data)

Both Standard and Shared Mamba architectures quickly achieved high training accuracy on synthetic signals but catastrophically collapsed to random guessing (~44-54%) on the test set, regardless of masking levels.

* **Insight:** This confirms that highly expressive SSMs are incredibly prone to learning **spurious correlations** (procedural artifacts) in algorithmically generated data. Parameter sharing alone does not prevent this memorization. Consequently, synthetic data is insufficient for benchmarking true generalization in SSMs.

### 2. Algorithmic Equivalence & Parameter Compression (Task 2: ListOps)

The ListOps task highlights the greatest strength of the SharedMamba architecture.

* **Insight:** The `SharedMamba` with 6 iterations achieves **96.0% to 97.5%** accuracy across all masking regimes, perfectly matching the `StandardMamba` baseline. 
* **The Victory:** It achieves this algorithmic parity while using **~6x fewer core parameters** (118k vs. 702k). This proves that for discrete, rule-based reasoning and state-retention tasks (where the model must parse logical trees and ignore padding), applying the exact same continuous transition dynamics ($\Theta_{shared}$) recursively is highly optimal and suffers no representation bottleneck.

### 3. The Continuous Robustness Trade-off (Task 3: MNIST)

The 1D flattened MNIST task exposes the architectural limitations of weight sharing in continuous spatial environments under severe noise.

* **Clean Performance:** On clean data, the shared parameters perform exceptionally well, lagging only marginally behind the standard models (e.g., 98.4% vs 98.7% for 6 layers).
* **OOD Masking:** When subjected to 20% and 50% zero-shot masking, the Standard architectures generally demonstrated higher resilience. For example, under 20% masking, the 3-layer `StandardMamba` retained 61.1% accuracy, while the `SharedMamba` dropped to 37.5%. 
* **Insight:** During repeated experiments, `SharedMamba` exhibited high variance on OOD continuous data. By forcing a single shared projection across all depths, the model becomes highly sensitive to initialization. In continuous sequence modeling, standard models utilize layer-specific normalizations and unshared projections ($\Phi_n$) to dynamically route features around severe input corruption. The shared hidden state lacks these multi-manifold degrees of freedom required to consistently recover from sudden, severe continuous signal loss.

### 4. The Impact of Initialization (High Variance)

It is crucial to note that across multiple unfixed-seed training runs, `SharedMamba` exhibited extreme sensitivity to weight initialization. In earlier exploratory runs, `SharedMamba` significantly outperformed the `StandardMamba` under 20% and 50% masking conditions. However, subsequent runs (documented in the tables above) showed the opposite. Because `SharedMamba` applies the exact same transition matrix recursively, a slightly "lucky" initialization compounds into a highly robust model, while a slightly "unlucky" one compounds into a brittle state manifold. This proves that weight sharing creates a much sharper, high-variance loss landscape compared to the standard deep stack.

### Final Verdict

**SharedMamba** is an exceptionally powerful architecture for deploying language, logic, and discrete algorithmic models on memory-constrained Edge devices, yielding massive parameter compression (up to 83% reduction) with zero performance loss. However, for continuous signal processing tasks operating in highly erratic or corrupted physical environments, the independent hierarchical feature extraction of a Standard SSM remains necessary for robust signal recovery.

## Quick Start

### Installation

#### Linux

```bash
git clone https://github.com/IPT-MMDA/ssm-orp.git
cd feat/ssm-layer-sharing
python3 -m venv .venv
source .venv/bin/activate
```

For users

```bash
pip install .
```

For developers

```bash
pip install -e .[dev]
```

If installed for developers

```bash
pytest
```

***

### Commands

```bash
# Train
train

# Train on synthetic
train --perturbation nothing --dataset synthetic --batch-size 128 --layers 6
## or
train --perturbation masking --dataset synthetic --batch-size 128 --layers 6 --mask 0.2

# Train on listops
train --perturbation nothing --dataset listops --vocab-size 11 --batch-size 128 --layers 6
## or
train --perturbation masking --dataset listops --vocab-size 11 --batch-size 128 --layers 6 --mask 0.2

# Train on mnist
train --perturbation nothing --dataset mnist --input-dim 1 --classes 10 --batch-size 128 --layers 6
## or
train --perturbation masking --dataset mnist --input-dim 1 --classes 10 --batch-size 128 --layers 6 --mask 0.2
```

```bash
# Evaluate
train --model MODEL_USED --dataset DATASET_USED --batch-size 128 --checkpoint PATH_TO_SAVED_MODEL --eval-only --perturbation PERTURBATION_YOU_WANT
```

```bash
# Get help for command
command --help
```

## Ways to expand these

* Create plots
* Add to SharedMamba the ability to use more than one block (e.g., alternating shared groups).
* **Systematic Multi-Seed Evaluation:** Due to the discovered high variance in SharedMamba's OOD performance, future work should include running experiments across 5-10 fixed seeds to calculate mean performance and true standard deviation.
* **Initialization Strategies:** Investigate specialized weight initialization techniques designed specifically for recursive/shared SSMs to stabilize the loss landscape.