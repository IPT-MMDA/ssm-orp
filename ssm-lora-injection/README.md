# Custom LoRA Adaptation for Mamba SSM: Domain Overfitting & OOD Evaluation

**Authors:** Serhii Poshtarenko and Dmytro Ishchenko

## Goal
Determine the generalization capacity and domain-shift robustness of a Mamba-based State Space Model (SSM) when fine-tuned using a highly restricted Parameter-Efficient Fine-Tuning (PEFT) method. We seek to prove that adapting *only* the input/output projections ($B$, $C$) while keeping the core transition dynamics ($A$) strictly frozen leads to severe domain overfitting.

## Methodology

### Dataset
The experiment is conducted on binary sentiment classification tasks, utilizing a primary domain and an Out-of-Distribution (OOD) domain to measure generalization:
* **In-Domain (IMDB):** Movie reviews. Used for training (400 samples) and baseline validation.
* **Out-of-Distribution (Yelp):** Restaurant reviews. Used exclusively to test the zero-shot domain transfer capabilities of the fine-tuned model.

### Model
Mamba Classifier: A classification head built on top of the pre-trained `mamba-130m-hf` backbone.
```text
input  (B, L)
  |
  [MambaBlock] x 24
    -> x_proj (dt, B, C)  <-- Target for LoRA
    -> A_log (transition) <-- Frozen
  |
  Mean Pooling
  |
  Linear(d_model -> 2)
  |
output (B, 2)
```
Default parameters: `d_model=768`, `n_layers=24`.

### Custom LoRA Injection
Unlike standard Transformer PEFT, which targets `Q/K/V` matrices, we perform architectural surgery on the SSM parameters:
1.  Target the `x_proj` layer responsible for generating data-dependent parameters ($B$, $C$, $\Delta t$).
2.  Inject a custom `LoRALinear` module (`Rank = 4`, `Alpha = 8`).
3.  **Strictly freeze** all original model weights, including the critical `A_log` eigenvalue parameter.
* **Parameter Efficiency:** The model contains ~130M parameters in total. Our custom LoRA injects only ~156K trainable parameters (**0.121%** of the total network).
* **Stability Hack:** Freezing $A$ places extreme load on the $B$ and $C$ projections, initially causing gradient instability (`NaN` loss). This was mitigated by reducing the Learning Rate to `5e-4` and applying Gradient Clipping (`max_norm=1.0`).

### Statistical Validation (Bootstrapping)
To ensure the reliability of the accuracy drop and variance, testing is performed using a **Bootstrapping** approach. The trained model in `eval()` mode is evaluated against 3 random subsets (100 samples each, `seeds: [42, 123, 999]`) for both IMDB and Yelp. A 95% Confidence Interval is calculated using Student's t-distribution.

## Results

All reported values are derived from the 3-seed bootstrapping evaluation (Mean ± 95% CI).

### Parameter Efficiency & Hardware Context
Theoretical Full Fine-Tuning (FFT) of the 130M parameters requires exponentially more VRAM to store gradients and AdamW optimizer states, exceeding standard consumer GPU limits. Our custom LoRA method successfully fits the training process within a <4GB VRAM footprint while achieving convergence in just 3-4 epochs, highlighting the necessity of PEFT for SSMs on local hardware.

### Domain Shift Tolerance

| Setting | Dataset / Domain | Mean Accuracy | 95% Confidence Interval |
| :--- | :--- | :--- | :--- |
| **In-Domain** | IMDB (Movies) | 74.33% | ± 3.79% (70.54% - 78.13%) |
| **OOD** | Yelp (Restaurants) | 66.67% | ± 7.59% (59.08% - 74.26%) |

Accuracy drops by ~7.6% when transitioning to the OOD dataset. There is a clear degradation in predictive power.

### Variance & Robustness
A counter-intuitive but mathematically sound result emerges from the confidence intervals:
* On the **In-Domain** data, the variance is relatively low (± 3.79%). The projections successfully learned to filter and interpret the movie-specific vocabulary.
* On the **OOD** data, the variance effectively doubles (± 7.59%). The upper bound of the Yelp CI (74.26%) lies below the baseline mean of IMDB.

**Key finding:** Freezing the state matrix $A$ forces the $B$ and $C$ matrices to overfit the domain vocabulary. When the context changes (from "plot/actor" to "menu/waiter"), the fundamental memory mechanism ($A$) cannot adapt to the new concepts. The model relies on overfitted projection filters, leading to higher uncertainty and degraded generalization.

## Limitations
* **Small Training Subset:** The training phase was restricted to 400 samples due to time/compute constraints. A full dataset run would likely yield higher absolute accuracies (e.g., 85%+), though the relative Domain Shift drop would remain.
* **Low Rank:** The chosen rank ($r=4$) represents an extreme bottleneck. Increasing the rank to 16 or 32 might recover some generalization capabilities without unfreezing $A$.
* **Single OOD Benchmark:** The generalization hypothesis was tested against only one distinct domain (Yelp). 

## How to run

### Google Colab (Recommended)
1. Open `data_analysis_lab.ipynb` in Colab.
2. Set runtime: `Runtime -> Change runtime type -> T4 GPU` (or better).
3. Run all cells sequentially. The notebook handles dependencies, training, and the final bootstrapping statistical test.

### Locally (Linux / Windows with CUDA)
Requires Python 3.9+ and a CUDA-compatible GPU (e.g., RTX 3060 or higher).

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies via pyproject.toml
pip install .

# 3. Run the project (if using modular structure)
python main.py
```

## File structure
```text
mamba-lora-ssm/
  README.md                  # This file
  pyproject.toml             # Dependency declaration
  data_analysis_lab.ipynb    # Main experimental notebook
  tests/                     # Unit tests
    test_architecture.py     # Verifies tensor shapes and math
    test_peft.py             # Verifies gradient freezing logic
    test_statistics.py       # Verifies Student's t-distribution math
  test_results/              # Generated test logs
