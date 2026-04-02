# SharedMamba: ALBERT-style Weight Sharing for State Space Models

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c.svg)

## Overview

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

***

### Commands

```bash
# Train
train

# Train on sythetic
train --perturbation nothing --dataset synthetic --batch-size 128

# Train on listops
train --perturbation nothing --dataset listops --vocab-size 11 --batch-size 128
## or
train --perturbation masking --dataset listops --vocab-size 11 --batch-size 128 --mask 0.2

# Train on mnist
train --perturbation nothing --dataset mnist --input-dim 1 --classes 10 --batch-size 128
## or
train --perturbation masking --dataset mnist --input-dim 1 --classes 10 --batch-size 128 --mask 0.2
```

```bash
# Evaluate
train --model MODEL_USED --dataset DATASET_USED --batch-size 128 --checkpoint PATH_TO_SAVED_MODEL --eval-only --perturbation PERTURBATION_YOU_WANT
```

```bash
# Get help for command
command --help
```

## План на останній день

*Моя задача:* Написати вичерпний звіт. Математичне обґрунтування через LaTeX, зведена таблиця кількості параметрів, таблиця з метриками (Accuracy на чистих даних vs Accuracy на зашумлених).

- (Опціонально) Зробити розподіленій не один лейр, а декілька
- Написати повністю README
- Покрити тестами
- Зробити графіки
