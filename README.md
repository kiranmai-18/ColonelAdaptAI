# ColonelAdaptAI: Prototype-Guided Continual Learning with Informative Perturbation Networks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20703196.svg)](https://doi.org/10.5281/zenodo.20703196)

## Overview

**ColonelAdaptAI** is a modular PyTorch framework for **continual learning** that combines prototype-guided memory, task-aware perturbation learning, and stability-preserving regularization to enable neural networks to learn sequential tasks while minimizing catastrophic forgetting.

The framework introduces **InfoPerturbNet**, a novel perturbation-based architecture that dynamically adapts internal feature representations through informative task-conditioned perturbations while preserving previously acquired knowledge.

The project provides a complete experimental pipeline for evaluating continual learning algorithms on standard benchmarks including:

* Split MNIST
* Permuted MNIST
* Split CIFAR-100

In addition, the framework includes several widely used rehearsal-based continual learning baselines for fair benchmarking and comparison.

---

# Key Features

### Continual Learning Framework

* Sequential task learning without retraining from scratch
* Multi-task incremental evaluation pipeline
* Catastrophic forgetting mitigation
* Stability-plasticity balancing mechanisms

### InfoPerturbNet Architecture

* Layer-wise informative perturbation modules
* Task-aware gating network
* Task encoder representation learning
* Adaptive feature modulation
* Shared backbone with task-conditioned adaptation

### Knowledge Preservation

* Prototype memory buffer
* Feature-space class prototype tracking
* Elastic Weight Consolidation (EWC)
* Fisher Information Matrix estimation
* Prototype-guided representation refinement

### Regularization Components

* EWC Stability Loss
* Prototype Refinement Loss
* Trust-Region Perturbation Loss
* Optional Contrastive Regularization

### Benchmark Support

* Split MNIST
* Permuted MNIST
* Split CIFAR-100

### Rehearsal-Based Baselines

* Fine-Tuning
* Replay
* GEM
* A-GEM
* DER
* DER++
* MIR

### Evaluation Metrics

* Average Accuracy
* Forgetting Measure
* Backward Transfer (BWT)
* Accuracy Matrix
* Per-task Performance Tracking

### Visualization & Reporting

* Accuracy curves
* Forgetting analysis plots
* Accuracy matrix heatmaps
* CSV result exports
* Automatic experiment logging

---

# Architecture

```text
Input
  │
  ▼
Shared CNN Backbone
  │
  ├───────────────────────┐
  │                       │
  ▼                       ▼
Task Encoder        Prototype Memory
  │                       │
  ▼                       │
Task-Aware Gating         │
  │                       │
  ▼                       │
InfoPerturb Units ◄───────┘
  │
  ▼
Refined Features
  │
  ▼
Task-Specific Heads
  │
  ▼
Predictions
```

---

# Project Structure

```text
ColonelAdaptAI/
│
├── main.py
├── config.yaml
│
├── data/
│   ├── datasets.py
│   └── task_generators.py
│
├── models/
│   ├── backbone.py
│   ├── infoperturbnet.py
│   ├── task_encoder.py
│   └── gating.py
│
├── memory/
│   ├── prototype_buffer.py
│   └── fisher_memory.py
│
├── losses/
│   ├── ewc_loss.py
│   ├── prototype_loss.py
│   └── trust_region_loss.py
│
├── trainers/
│   └── continual_trainer.py
│
├── baselines/
│   ├── finetune.py
│   ├── replay.py
│   ├── agem.py
│   ├── gem.py
│   ├── der.py
│   ├── derpp.py
│   └── mir.py
│
├── utils/
│   ├── metrics.py
│   ├── plotting.py
│   └── logging_utils.py
│
└── results/
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/kiranmai-18/ColonelAdaptAI

cd ColonelAdaptAI
```

## Create Environment

```bash
conda create -n coloneladaptai python=3.10

conda activate coloneladaptai
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Quick Start

## Split MNIST

```bash
python main.py \
    --dataset split_mnist \
    --epochs 2 \
    --batch-size 128 \
    --output-dir results/split_mnist_run
```

---

## Permuted MNIST

```bash
python main.py \
    --dataset permuted_mnist \
    --epochs 1 \
    --batch-size 128 \
    --output-dir results/permuted_mnist_run
```

---

## Split CIFAR-100

```bash
python main.py \
    --dataset split_cifar100 \
    --epochs 5 \
    --batch-size 128 \
    --output-dir results/split_cifar100_run
```

---

# Running Continual Learning Baselines

## Fine-Tuning

```bash
python main.py \
    --dataset split_mnist \
    --method finetune \
    --epochs 2
```

## Replay

```bash
python main.py \
    --dataset split_mnist \
    --method replay \
    --epochs 2
```

## A-GEM

```bash
python main.py \
    --dataset split_mnist \
    --method agem \
    --epochs 2
```

## GEM

```bash
python main.py \
    --dataset split_mnist \
    --method gem \
    --epochs 2
```

## DER

```bash
python main.py \
    --dataset split_mnist \
    --method der \
    --epochs 2
```

## DER++

```bash
python main.py \
    --dataset split_mnist \
    --method derpp \
    --epochs 2
```

## MIR

```bash
python main.py \
    --dataset split_mnist \
    --method mir \
    --epochs 2
```

---

## Run Complete Baseline Suite

```bash
python main.py \
    --dataset split_mnist \
    --run-baselines \
    --epochs 2 \
    --output-dir results/baseline_suite
```

---

# Configuration

All experiment settings can be modified in:

```text
config.yaml
```

Example:

```yaml
training:
  epochs: 5
  batch_size: 128
  learning_rate: 0.001

loss:
  ewc_lambda: 10.0
  proto_lambda: 0.2
  trust_lambda: 0.01
  contrastive_lambda: 0.0

memory:
  prototype_size: 2000
  fisher_samples: 1024
```

---

# Baseline Hyperparameters

```yaml
baselines:
  run: false
  methods:
    - finetune
    - replay
    - agem
    - gem
    - der
    - derpp
    - mir

  memory_size: 2000
  replay_batch_size: 128

  der_alpha: 0.5
  der_beta: 0.5

  mir_candidates: 64
  mir_k: 128
```

---

# Output Files

Each experiment automatically generates:

```text
results/
│
├── accuracy_matrix.csv
├── summary_results.csv
├── accuracy_matrix.png
├── avg_accuracy.png
├── forgetting.png
├── checkpoint.pt
├── used_config.json
└── log.txt
```

---

# Evaluation Metrics

### Average Accuracy (ACC)

Measures final average performance across all tasks.

### Forgetting Measure (FM)

Quantifies performance degradation on previously learned tasks.

### Backward Transfer (BWT)

Measures the influence of learning new tasks on old tasks.

### Accuracy Matrix

Stores task-wise accuracy throughout the continual learning process.

---

# Method Summary

InfoPerturbNet introduces lightweight perturbation modules that adapt feature representations according to task context while preserving previously learned knowledge.

The framework combines:

* Task-aware perturbation learning
* Prototype-guided feature refinement
* Fisher-based parameter consolidation
* Trust-region regularization

This combination enables stable continual adaptation while maintaining high plasticity for new tasks.

---

# Experimental Benchmarks

| Dataset         | Tasks |
| --------------- | ----- |
| Split MNIST     | 5     |
| Permuted MNIST  | 10    |
| Split CIFAR-100 | 10    |

---

# Future Extensions

* Vision Transformers (ViT)
* CLIP-based continual learning
* Dynamic prototype compression
* Federated continual learning
* Online task discovery
* Open-world continual learning
* Multi-modal continual adaptation

---

# Citation

```bibtex
@software{ColonelAdaptAI2026,
  title={ColonelAdaptAI: Prototype-Guided Continual Learning with Informative Perturbation Networks},
  year={2026},
}
```

---

# License

This project is released under the MIT License.

