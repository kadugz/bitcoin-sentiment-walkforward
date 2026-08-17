"""Gera 09_predicted_vs_actual.png: fechamento real vs. previsto (t+1), costurado nos 5
folds walk-forward, com RMSE de cada modelo anotado no gráfico."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from paper_style import apply_paper_style, PALETTE
from lstm_utils import (
    CONFIG, FEATURE_COLS_TECH, FEATURE_COLS_SENT, FIGS_DIR, DATA_PROCESSED,
    walk_forward_splits, fit_scaler_and_transform, build_sequences, train_lstm,
)

apply_paper_style()


def make_predicted_vs_actual():
    print("=" * 70)
    print("Figura 09: previsto vs. real (walk-forward, fora da amostra)")
    print("=" * 70)

    dataset = pd.read_csv(DATA_PROCESSED / "dataset_features_target.csv", index_col=0, parse_dates=True)

    feature_sets = {
        "numeric_only": FEATURE_COLS_TECH,
        "numeric_plus_sentiment": FEATURE_COLS_TECH + FEATURE_COLS_SENT,
    }

    folds = walk_forward_splits(dataset.index, CONFIG["n_walk_forward_folds"], CONFIG["min_train_days"])
    seq_len = CONFIG["sequence_length"]
    seed = CONFIG["seed"]

    stitched = {name: [] for name in feature_sets}
    actual_rows = []

    for fold in folds:
        idx_te = None
        for name, cols in feature_sets.items():
            scaler, scaled = fit_scaler_and_transform(fold["train"], dataset, cols)
            X_tr, yreg_tr, yclf_tr, _ = build_sequences(scaled, fold["train"], seq_len, dataset)
            X_val, yreg_val, yclf_val, _ = build_sequences(scaled, fold["val"], seq_len, dataset)
            X_te, yreg_te, yclf_te, idx_te = build_sequences(scaled, fold["test"], seq_len, dataset)

            model = train_lstm(X_tr, yreg_tr, yclf_tr, X_val, yreg_val, yclf_val, len(cols), seed)
            with torch.no_grad():
                pred_reg, _ = model(torch.tensor(X_te))
            pred_reg = pred_reg.numpy()
            close_t = dataset.loc[idx_te, "close"].values
            pred_close = close_t * np.exp(pred_reg)

            target_date = idx_te + pd.Timedelta(days=1)
            stitched[name].append(pd.Series(pred_close, index=target_date))
        print(f"  fold {fold['fold']} treinado (numeric_only + numeric_plus_sentiment), n_test={len(idx_te)}")

        actual_close = dataset.loc[idx_te, "target_close"].copy()
        actual_close.index = idx_te + pd.Timedelta(days=1)
        actual_rows.append(actual_close)

    actual_series = pd.concat(actual_rows).sort_index()
    pred_series = {name: pd.concat(parts).sort_index() for name, parts in stitched.items()}

    rmse = {
        name: float(np.sqrt(np.mean((series.values - actual_series.values) ** 2)))
        for name, series in pred_series.items()
    }
    mae = {
        name: float(np.mean(np.abs(series.values - actual_series.values)))
        for name, series in pred_series.items()
    }
    delta_pct = (rmse["numeric_plus_sentiment"] / rmse["numeric_only"] - 1.0) * 100
    direction = "piora" if delta_pct > 0 else "melhora"

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(actual_series.index, actual_series.values, color=PALETTE["real"], label="Real")
    ax.plot(pred_series["numeric_only"].index, pred_series["numeric_only"].values,
            color=PALETTE["primary"], linewidth=1.0, alpha=0.85, label="Previsto — numérico puro")
    ax.plot(pred_series["numeric_plus_sentiment"].index, pred_series["numeric_plus_sentiment"].values,
            color=PALETTE["sentiment"], linewidth=1.0, alpha=0.85, label="Previsto — numérico + sentimento")
    ax.set_title("Fechamento real vs. previsto (t+1)")
    ax.set_ylabel("Preço (USD)")

    stats_text = (
        f"RMSE numérico puro: {rmse['numeric_only']:.0f} USD\n"
        f"RMSE numérico + sentimento: {rmse['numeric_plus_sentiment']:.0f} USD\n"
        f"Sentimento {direction} o RMSE em {abs(delta_pct):.1f}% (seed=0, único fold-set)"
    )
    ax.text(0.01, 0.02, stats_text, transform=ax.transAxes, fontsize=8.5,
            va="bottom", ha="left", color="#333333",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc", alpha=0.9))

    ax.legend(loc="upper left")
    plt.tight_layout()
    out_path = FIGS_DIR / "09_predicted_vs_actual.png"
    plt.savefig(out_path)
    plt.close(fig)
    print("Figura salva em:", out_path)
    print(f"  numeric_only: RMSE={rmse['numeric_only']:.2f} MAE={mae['numeric_only']:.2f}")
    print(f"  numeric_plus_sentiment: RMSE={rmse['numeric_plus_sentiment']:.2f} MAE={mae['numeric_plus_sentiment']:.2f}")
    print("  (comparar com outputs/final_comparison_table.csv para a métrica oficial agregada em 5 seeds x 5 folds)")


if __name__ == "__main__":
    make_predicted_vs_actual()
    print("\nConcluído.")
