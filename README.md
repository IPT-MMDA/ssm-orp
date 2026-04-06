# Task: Implement and evaluate 8-bit integer (INT8) quantization specifically for the transition and projection matrices (A, B, C) of a pre-trained SSM. Measure the robustness degradation vs the standard.

## About this project:
This project investigates the impact of Post-Training Quantization (PTQ) on State Space Models (SSMs), with a focus on reducing the precision of the core parameters:
- Transition matrix A
- Input projection matrix B
- Output projection matrix C

The main objective is to analyze how quantization to INT8 precision affects:
- model performance
- robustness under perturbations
- theoretical memory efficiency

## Implementation
I used simplified SSM model:
h_{t+1} = A h_t + B x_t
y_t = C h_t

The dataset is synthetic: sequences are generated using another random (stable) SSM

### Structure:

model.py         # SSM model
data.py          # synthetic dataset
train.py         # training loop
eval.py          # evaluation
quant_utils.py   # quantization logic
benchmark.py     # timing, memory
main.py          # entry point

### How to run:
