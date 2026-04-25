# SSM Knowledge Distillation

## Goal

Map how Knowledge Distillation (KD) hyperparameters (temperature T and loss weight alpha) affect the accuracy and robustness of a miniaturized student SSM (Mamba) trained on Speech Commands v2 (35 classes, raw 16kHz waveforms).

## Architecture

Both teacher and student use the same structure: strided Conv1d (stride=16, compresses 16k samples to 1k steps) followrd by Mamba blocks with pre-norm residuals and global average pooling into a classification head.

| Model | d_model | layers | approx params |
|---|---|---|---|
| Teacher | 128 | 6 | 600K |
| Large student | 64 | 4 | 150K |
| Small student | 32 | 2 | 20K |

## KD loss

$$
\mathcal{L} = \alpha \cdot T^2 \cdot D_{KL}\!\left(\text{softmax}\!\left(\frac{z_s}{T}\right) \;\middle\|\; \text{softmax}\!\left(\frac{z_t}{T}\right)\right) + (1 - \alpha) \cdot \mathcal{L}_{CE}(z_s, y)
$$

where $z_s$ and $z_t$ are the student and teacher logits, $T$ is the temperature, $\alpha$ is the balance factor, and $y$ are the hard labels.

- $\alpha = 0$: no distillation, pure hard labels
- $\alpha = 1$: pure soft targets from teacher
- Higher $T$ softens the teacher distribution, revealing more inter-class relatioships

## Grid search

- T in {1, 2, 5, 10}
- alpha in {0.0, 0.5, 1.0}
- 2 seeds per config (42, 123)
- 10 epochs per student, AdamW lr=3e-3, cosine annealing
- Teacher logits cached before grid search (single forward pass)

## Project structure

```
feat/ssm-knowledge-distillation/
    README.md
    pyproject.toml
    analysis.ipynb                          # plots and analysis
    distillation_results.csv                # large student results
    distillation_results_small_student.csv  # small student results
    robustness_results.csv                  # large student robustness
    robustness_results_small_student.csv    # small student robustness
    .gitignore
    tools/
        download_dataset.py                 # download Speech Commands v2
    scripts/
        dataset.py          # shared dataset class with .pt caching
        teacher.py           # TeacherSSM model definition
        student.py           # StudentSSM model definition
        train_teacher.py     # train the teacher model
        distill.py           # KD grid search
        evaluate_robustness.py  # robustness evaluation suite
        test_models.py       # unit tests
    dataset/                 # raw data (gitignored)
    models/                  # saved checkpoints (gitignored)
```

## How to run

Requires Linux/WSL, CUDA GPU, Python 3.10+.

1. create venv and install dependencies

2. download dataset with tools/download_dataset.py

3. train teacher scripts/train_teacher.py. All scripts cache dataset as .pt for faster loads, if it breaks, it will just load dataset from disk.

4. run distillation grid search scripts/distill.py. It saves student checkpoints to models/ and results to distillation_results.csv, resumes from CSV if interrupted

5. evaluate robustness scripts/evaluate_robustness.py. Saves robustness_results.csv, resumes if interrupted. Use --small flag for small student models

6. run tests scripts/test_models.py

7. view analysis.ipynb in Jupyter/VS Code

## Results (obtained by M.Holub, creator of the feat/ssm-knowledge-distilation)

### Teacher
- 94.5% accuracy, 94.2% macro F1 on validation (15 epochs)

### Small student (d=32, 2 layers)

| T  | alpha=0.0 | alpha=0.5 | alpha=1.0 |
|---|---|---|---|
| 1  | 73.0% | 74.0% | 73.6% |
| 2  | 73.0% | 73.1% | 71.4% |
| 5  | 74.0% | 70.9% | 71.5% |
| 10 | 73.9% | 73.0% | 71.9% |

Distillation did not help the small student. alpha=0.0 (no distillation) was competitive or best across all temperatures. The student is too small to absorb the teacher's soft knowledge. This actually means that KD isn't an universal improovement and can hurt small models (Please expose small models only to daycare, not teachers (this is a joke (obviously (or maybe not obviously))))

### Large student (d=64, 4 layers)

| T  | alpha=0.0 | alpha=0.5 | alpha=1.0 |
|---|---|---|---|
| 1  | 87.8% | 88.2% | 86.6% |
| 2  | 85.0% | 87.8% | 86.6% |
| 5  | 87.6% | 86.6% | 87.3% |
| 10 | 87.8% | 87.1% | 86.3% |

alpha=0.5 at T=1 gave the best result (88.2%). Distillation provides a small but consistent benefit at moderate alpha. Pure soft targets (alpha=1.0) tend to hurt.

### Robustness (large student)

Evaluated on test set with FGSM (eps 0.01/0.05/0.1), Gaussian noise (20/10/5 dB SNR), and sequence truncation (25/50/75%).

Key findings:
- **FGSM adversarial robustness increases with T and alpha.** T=10, alpha=1.0 achieves 82% under FGSM eps=0.1, vs 65% for alpha=0.0. Soft targets teach smoother decision bondaries.
- **Noise robustness decreases with T and alpha.** alpha=0.0 retains 47% at 5dB SNR, while T=10 alpha=1.0 drops to 27%. Distilled models become dependent on full signal structure.
- **Truncation robustness is roughly uniform** across configs (49% at 50% truncation, 14% at 75%). This is primarily architecture-dependent, not KD-dependent.

### Robustness (small student)

Same evaluation suite on the small student (d=32, 2 layers).

Key findings:
- **FGSM robustness also increases with T and alpha**, same pattern as large student. T=10, alpha=1.0 achieves 63% under FGSM eps=0.1, vs 24% for alpha=0.0. The effect is even more dramatic than the large student.
- **Noise robustness is much worse overall** due to the small model's lower capacity. At 5dB SNR, all configs are below 18%.
- **Truncation robustness is uniformly poor** (33% at 50%, 9% at 75%).

### Overall robustness conclusion

Robustness is a trade-off: distillation with high temperature trades noise robustness for adversarial robustness. This holds for both student sizes. The effect is amplified in smaller students. The optimal soluton will depend on the deployment scenario.

## Limitations of my (M. Holub) computations
- Have only done 2 seeds per config; confidence intervals are W I D E
- Alpha grid is coarse {0.0, 0.5, 1.0}; finer resolution (i.e. with 0.25, 0.75) may reveal a better sweet spot. Also Grid search is only useful for seeing patterns in params. To find good params PSO or any other fitness-finding algorythm will do much better and maybe even faster. 
- Student architectures are fixed; the capacity gap's effect on KD could be explored further
- No data augmentatoin was used during training
- WSL/NTFS filesystem bottlenecks required workarounds (dataset caching to Linux filesystem)
