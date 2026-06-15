from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_accuracy_matrix(matrix, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(np.asarray(matrix, dtype=float), aspect='auto')
    ax.set_xlabel('Evaluated Task')
    ax.set_ylabel('After Training Task')
    ax.set_title('Continual Learning Accuracy Matrix')
    fig.colorbar(im, ax=ax, label='Accuracy')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def plot_curve(values, ylabel, title, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(values)+1), values, marker='o')
    ax.set_xlabel('Task')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
