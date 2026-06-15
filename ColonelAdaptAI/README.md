# ColonelAdaptAI / InfoPerturbNet

A modular PyTorch implementation of **ColonelAdaptAI**, a prototype-guided perturbation learning framework for stable continual neural network adaptation, and its core model **InfoPerturbNet**.

## Implemented Components

- Split MNIST, Permuted MNIST, and Split CIFAR-100 task streams
- Shared CNN backbones for MNIST-style and CIFAR-style inputs
- InfoPerturbNet with layer-wise informative perturbation units
- Task encoder and task-aware gating module
- Task-specific classification heads
- Prototype buffer for feature-space class prototypes
- Fisher diagonal memory and EWC stability regularization
- Prototype-guided refinement loss
- Trust-region perturbation norm loss
- Continual learning metrics:
  - Average Accuracy
  - Forgetting Measure
  - Backward Transfer
  - Accuracy Matrix
- Result CSV files and plots
- Functional rehearsal baselines: Fine-Tuning, Replay, GEM, A-GEM, DER, DER++, and MIR

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Run Split MNIST:

```bash
python main.py --dataset split_mnist --epochs 2 --batch-size 128 --output-dir results/split_mnist_run
```

Run Permuted MNIST:

```bash
python main.py --dataset permuted_mnist --epochs 1 --batch-size 128 --output-dir results/permuted_mnist_run
```

Run Split CIFAR-100:

```bash
python main.py --dataset split_cifar100 --epochs 5 --batch-size 128 --output-dir results/split_cifar100_run
```


## Running Rehearsal-Based Baselines

Run one baseline at a time:

```bash
python main.py --dataset split_mnist --method agem --epochs 2 --output-dir results/split_mnist_agem
python main.py --dataset split_mnist --method gem --epochs 2 --output-dir results/split_mnist_gem
python main.py --dataset split_mnist --method der --epochs 2 --output-dir results/split_mnist_der
python main.py --dataset split_mnist --method derpp --epochs 2 --output-dir results/split_mnist_derpp
python main.py --dataset split_mnist --method mir --epochs 2 --output-dir results/split_mnist_mir
```

Run the full baseline suite listed in `config.yaml`:

```bash
python main.py --dataset split_mnist --run-baselines --epochs 2 --output-dir results/baseline_suite
```

Implemented baseline behavior:

| Method | Implementation Summary |
|---|---|
| Fine-Tuning | Current-task training without replay or stability regularization |
| Replay | Supervised replay using episodic memory samples |
| A-GEM | Projects current gradients against an averaged replay gradient when interference is detected |
| GEM | Applies task-wise replay-gradient projection using stored memory from previous tasks |
| DER | Uses logit-based replay distillation to preserve previous decision boundaries |
| DER++ | Combines DER logit replay with supervised replay cross-entropy |
| MIR | Selects replay samples with the largest expected post-update loss increase |

Baseline hyperparameters can be changed in `config.yaml`:

```yaml
baselines:
  run: false
  methods: [finetune, replay, agem, gem, der, derpp, mir]
  memory_size: 2000
  replay_batch_size: 128
  der_alpha: 0.5
  der_beta: 0.5
  mir_candidates: 64
  mir_k: 128
```

## Project Structure

```text
ColonelAdaptAI/
├── main.py
├── config.yaml
├── data/
├── models/
├── memory/
├── losses/
├── trainers/
├── baselines/
├── utils/
└── results/
```

## Main Configuration

Edit `config.yaml` to change dataset, epochs, learning rate, loss weights, Fisher sample count, and output folder.

Important loss weights:

```yaml
loss:
  ewc_lambda: 10.0
  proto_lambda: 0.2
  trust_lambda: 0.01
  contrastive_lambda: 0.0
```

Set all continual-learning regularizers to zero to obtain a simple fine-tuning baseline.

## Output Files

After training, the selected output directory contains:

```text
accuracy_matrix.csv
summary_results.csv
accuracy_matrix.png
avg_accuracy.png
forgetting.png
checkpoint.pt
used_config.json
log.txt
```

## Notes

This implementation makes practical engineering decisions to convert the methodology into a runnable research prototype. It keeps the prototype buffer in feature space, uses diagonal Fisher estimation for EWC, and injects compact task-layer perturbation vectors at backbone merge points. The baseline files now include functional rehearsal-based implementations of Replay, GEM, A-GEM, DER, DER++, and MIR for paper-level benchmarking under the same task stream and evaluation protocol.
