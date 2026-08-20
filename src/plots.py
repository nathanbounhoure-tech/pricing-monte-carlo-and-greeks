"""
plots.py

Génération des figures du projet avec matplotlib (backend non-interactif "Agg"),
exportées en PNG dans outputs/figures/.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
    }
)


def plot_convergence(df, bs_ref: float, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["n_paths"], df["price"], marker="o", color="#1f77b4", label="Prix Monte Carlo (brut)")
    ax.fill_between(
        df["n_paths"],
        df["price"] - 1.96 * df["std_error"],
        df["price"] + 1.96 * df["std_error"],
        alpha=0.2,
        color="#1f77b4",
        label="Intervalle de confiance à 95%",
    )
    ax.axhline(bs_ref, color="black", linestyle="--", linewidth=1.5, label="Black-Scholes (référence)")
    ax.set_xscale("log")
    ax.set_xlabel("Nombre de trajectoires N (échelle log)")
    ax.set_ylabel("Prix du call européen")
    ax.set_title("Convergence Monte Carlo vs Black-Scholes — call européen")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_variance_reduction_comparison(labels, widths_base, widths_reduced, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, widths_base, width, label="Sans réduction de variance", color="#d62728")
    ax.bar(x + width / 2, widths_reduced, width, label="Antithétique + variable de contrôle", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Largeur de l'intervalle de confiance à 95%")
    ax.set_title("Effet de la réduction de variance sur la précision statistique")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sensitivity_volatility(df, path: str) -> None:
    sub = df[df["parameter"] == "sigma"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sub["value"], sub["price"], marker="o", color="#9467bd")
    ax.set_xlabel("Volatilité (sigma)")
    ax.set_ylabel("Prix du call européen (Black-Scholes)")
    ax.set_title("Sensibilité du prix à la volatilité")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_simulated_paths_with_barrier(paths, times, barrier: float, path: str, n_show: int = 40) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    n_show = min(n_show, paths.shape[0])
    alive_color, knocked_color = "#1f77b4", "#d62728"
    for i in range(n_show):
        touched = paths[i].min() <= barrier
        ax.plot(
            times,
            paths[i],
            linewidth=0.9,
            alpha=0.75,
            color=knocked_color if touched else alive_color,
        )
    ax.axhline(barrier, color="black", linestyle="--", linewidth=2, label="Niveau de barrière")
    ax.set_xlabel("Temps (années)")
    ax.set_ylabel("Prix du sous-jacent")
    ax.set_title("Trajectoires simulées et barrière (down-and-out call)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
