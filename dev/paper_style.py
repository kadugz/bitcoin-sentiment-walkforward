"""Estilo visual compartilhado ("paper" clássico) das figuras do projeto."""
import matplotlib.pyplot as plt

# azul=numérico/técnico | vermelho=sentimento | verde=positivo | cinza=neutro/volume
PALETTE = {
    "primary": "#2E5C8A",
    "sentiment": "#B3483D",
    "positive": "#4C7A52",
    "negative": "#B3483D",
    "neutral": "#8C8C8C",
    "secondary": "#C97B3D",
    "accent": "#6B5B95",
    "real": "#1A1A1A",
}


def apply_paper_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Georgia", "DejaVu Serif", "Times New Roman", "serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "lines.linewidth": 1.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
    })
