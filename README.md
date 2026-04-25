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

The dataset is synthetic: sequences are generated using another random (stable) SSM.

I implemented weight-only PTQ for matrices A, B, and C. The quantization pipeline computes scale and zero-point values, then performs quantization to INT8 followed by dequantization back to float. 

The model is evaluated under several conditions: clean test data, Gaussian noise, sequence shift perturbations.

## Structure:
```bash
ssm_ptq_project/
├── train_baseline.py # train FP32 baseline model
├── run_ptq_experiment.py # run PTQ and evaluation
├── plot_results.py # generate plots from results

├── model.py # SSM model
├── data.py # dataset generation
├── config.py # configuration (data, training, quantization)
├── utils.py # general utilities 

├── quant_utils.py # quantization logic 
├── eval_utils.py # evaluation metrics 
├── robustness_utils.py # noise and perturbation functions
├── benchmark_utils.py # memory and latency measurement

├── artifacts/ # saved models, results, plots
```

## How to run:
(anaconda prompt)

Create environment
```bash
conda create -n ssm_ptq python=3.10 -y
conda activate ssm_ptq
pip install torch numpy pandas scikit-learn matplotlib pytest
```

(or activate it:

```bash
conda activate ssm_ptq
```
)

Navigate to the project directory
```bash
cd ~\ssm_ptq_project
```

```bash

python train_baseline.py

python run_ptq_experiment.py

python plot_results.py
```
