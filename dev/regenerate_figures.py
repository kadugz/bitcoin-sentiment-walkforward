"""Regenera as figuras 01-08 no estilo paper (dev/paper_style.py), reaproveitando dados
cacheados; só a fig. 08 (trading) precisa retreinar o LSTM, pois a curva diária não é salva."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose

from paper_style import apply_paper_style, PALETTE
from lstm_utils import (
    CONFIG, FEATURE_COLS_TECH, FEATURE_COLS_SENT, FIGS_DIR, DATA_RAW, DATA_PROCESSED,
    OUTPUTS_DIR, walk_forward_splits, fit_scaler_and_transform, build_sequences, train_lstm,
)

apply_paper_style()

price_df = pd.read_csv(DATA_RAW / "btc_usd_ohlcv.csv", index_col=0, parse_dates=True)
dataset = pd.read_csv(DATA_PROCESSED / "dataset_features_target.csv", index_col=0, parse_dates=True)
FEATURE_COLUMNS = FEATURE_COLS_TECH + FEATURE_COLS_SENT


def load_news_with_sentiment() -> pd.DataFrame:
    """Reconstrói `news` (com sent_score/sent_class) a partir dos caches, sem rechamar o FinBERT."""
    news_raw = pd.read_csv(DATA_RAW / "cryptonews.csv")
    news = news_raw.copy()
    news["date"] = pd.to_datetime(news["date"], errors="coerce", utc=True)
    news = news.dropna(subset=["date"]).copy()
    news["news_date"] = news["date"].dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None)

    text_field = "title"
    news = news.dropna(subset=[text_field]).copy()
    news[text_field] = news[text_field].astype(str).str.strip()
    news = news[news[text_field].str.len() > 0].copy()

    sent_df = pd.read_csv(DATA_PROCESSED / "news_finbert_sentiment.csv")
    assert len(sent_df) == len(news), "Cache de sentimento não bate com news limpo — verifique os caches."
    news = news.reset_index(drop=True)
    sent_df = sent_df.reset_index(drop=True)
    return pd.concat([news, sent_df], axis=1)


def fig01_price_overview():
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(price_df.index, price_df["Close"], color=PALETTE["primary"])
    axes[0].set_title("BTC-USD — preço de fechamento diário")
    axes[0].set_ylabel("Preço (USD)")

    axes[1].bar(price_df.index, price_df["Volume"], color=PALETTE["neutral"], width=1.0)
    axes[1].set_title("Volume diário")
    axes[1].set_ylabel("Volume")

    plt.tight_layout()
    path = FIGS_DIR / "01_price_overview.png"
    plt.savefig(path)
    plt.close(fig)
    print("Salvo:", path)


def fig02_news_coverage(news: pd.DataFrame):
    news_per_day = news.groupby("news_date").size()
    full_range = pd.date_range(price_df.index.min(), price_df.index.max(), freq="D")
    news_per_day_full = news_per_day.reindex(full_range, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(news_per_day_full.index, news_per_day_full.values, color=PALETTE["positive"], alpha=0.55)
    ax.axvspan(price_df.index.min(), news_per_day.index.min(), color=PALETTE["negative"], alpha=0.08,
               label="sem cobertura de notícias")
    ax.axvspan(news_per_day.index.max(), price_df.index.max(), color=PALETTE["negative"], alpha=0.08)
    ax.set_title("Notícias por dia vs. histórico de preço")
    ax.set_ylabel("Notícias/dia")
    ax.legend(loc="upper right")
    plt.tight_layout()
    path = FIGS_DIR / "02_news_coverage.png"
    plt.savefig(path)
    plt.close(fig)
    print("Salvo:", path)


def fig03_sentiment_distribution(news: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(news["sent_score"], bins=50, color=PALETTE["accent"], ax=axes[0])
    axes[0].set_title("Distribuição do score de sentimento")
    axes[0].set_xlabel("sent_score = P(pos) − P(neg)")

    news["sent_class"].value_counts().reindex(["positive", "neutral", "negative"]).plot(
        kind="bar",
        color=[PALETTE["positive"], PALETTE["neutral"], PALETTE["negative"]],
        ax=axes[1],
    )
    axes[1].set_title("Contagem por classe de sentimento")
    axes[1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    path = FIGS_DIR / "03_sentiment_distribution.png"
    plt.savefig(path)
    plt.close(fig)
    print("Salvo:", path)


def fig03b_seasonality():
    seasonal_periods = [1, 30, 365]
    log_close = np.log(price_df["Close"]).asfreq("D")

    results, rows = {}, []
    for period in seasonal_periods:
        r = seasonal_decompose(log_close, model="additive", period=period, extrapolate_trend="freq")
        resid_var = np.nanvar(r.resid)
        total_var = np.nanvar(r.resid + r.seasonal)
        strength = 0.0 if total_var == 0 else max(0.0, 1 - resid_var / total_var)
        results[period] = r
        rows.append({"period_days": period, "seasonal_strength": strength})
    strength_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(len(seasonal_periods), 3, figsize=(14, 8), sharex=True)
    for i, period in enumerate(seasonal_periods):
        r = results[period]
        strength = strength_df.loc[strength_df["period_days"] == period, "seasonal_strength"].iloc[0]
        axes[i, 0].plot(r.trend, color=PALETTE["primary"], linewidth=1)
        axes[i, 0].set_ylabel(f"period={period}d\n\nTendência")
        axes[i, 1].plot(r.seasonal, color=PALETTE["positive"], linewidth=0.8)
        axes[i, 1].set_ylabel("Sazonalidade")
        axes[i, 1].set_title(f"força sazonal = {strength:.3f}", fontsize=9)
        axes[i, 2].plot(r.resid, color=PALETTE["negative"], linewidth=0.6)
        axes[i, 2].set_ylabel("Resíduo")
    axes[0, 0].set_title("Tendência (log-preço)")
    axes[0, 1].set_title("Sazonalidade")
    axes[0, 2].set_title("Resíduo")
    fig.suptitle("BTC-USD — decomposição sazonal em três escalas", y=1.02, fontweight="bold")
    plt.tight_layout()
    path = FIGS_DIR / "03b_seasonality_multiscale.png"
    plt.savefig(path)
    plt.close(fig)
    print("Salvo:", path)


def fig04_pca_variance():
    X = dataset[FEATURE_COLUMNS].copy()
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(15, X_scaled.shape[1])).fit(X_scaled)
    explained = pca.explained_variance_ratio_
    cum_explained = np.cumsum(explained)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(range(1, len(explained) + 1), explained, color=PALETTE["primary"])
    axes[0].set_title("Variância explicada por componente")
    axes[0].set_xlabel("Componente")
    axes[0].set_ylabel("Proporção da variância")

    axes[1].plot(range(1, len(cum_explained) + 1), cum_explained, marker="o", color=PALETTE["secondary"])
    axes[1].axhline(0.9, color=PALETTE["neutral"], linestyle="--", linewidth=1)
    axes[1].set_title("Variância explicada acumulada")
    axes[1].set_xlabel("Nº de componentes")
    axes[1].set_ylabel("Acumulada")

    plt.tight_layout()
    path = FIGS_DIR / "04_pca_variance.png"
    plt.savefig(path)
    plt.close(fig)
    print("Salvo:", path)


def fig05_correlation_heatmap():
    corr_cols = [
        "log_return_1d", "volatility_7", "rsi_14", "price_to_ma_30",
        "sent_mean", "sent_prop_positive", "sent_prop_negative", "news_count", "has_news",
        "target_log_return",
    ]
    corr_matrix = dataset[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax, vmin=-1, vmax=1,
                linewidths=0.5, linecolor="white", cbar_kws={"shrink": 0.85})
    ax.set_title("Correlação entre features-chave e o alvo")
    plt.tight_layout()
    path = FIGS_DIR / "05_correlation_heatmap.png"
    plt.savefig(path)
    plt.close(fig)
    print("Salvo:", path)


def fig06_walk_forward_scheme(folds):
    fig, ax = plt.subplots(figsize=(11, 3.2))
    colors = {"train": PALETTE["primary"], "val": PALETTE["secondary"], "test": PALETTE["positive"]}
    for f in folds:
        y = f["fold"]
        for part in ["train", "val", "test"]:
            idx = f[part]
            ax.barh(y, (idx.max() - idx.min()).days + 1, left=idx.min(), height=0.6,
                    color=colors[part], label=part if y == 0 else None)
    ax.set_yticks([f["fold"] for f in folds])
    ax.set_yticklabels([f"Fold {f['fold']}" for f in folds])
    ax.set_title("Esquema walk-forward — treino / validação / teste")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    path = FIGS_DIR / "06_walk_forward_scheme.png"
    plt.savefig(path)
    plt.close(fig)
    print("Salvo:", path)


def fig07_permutation_importance():
    df = pd.read_csv(OUTPUTS_DIR / "permutation_importance.csv")
    plot_df = df.sort_values("delta_RMSE")
    colors_imp = [PALETTE["sentiment"] if s else PALETTE["primary"] for s in plot_df["is_sentiment_feature"]]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(plot_df["feature"], plot_df["delta_RMSE"], color=colors_imp)
    ax.set_xlabel("Aumento no RMSE ao embaralhar a feature")
    ax.set_title("Permutation importance")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["primary"], label="Feature técnica"),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["sentiment"], label="Feature de sentimento"),
    ]
    ax.legend(handles=handles, loc="lower right")
    plt.tight_layout()
    path = FIGS_DIR / "07_permutation_importance.png"
    plt.savefig(path)
    plt.close(fig)
    print("Salvo:", path)


def simulate_strategy(signal: np.ndarray, log_returns: np.ndarray, cost_pct: float):
    signal = np.asarray(signal, dtype=float)
    position_prev = np.concatenate([[0.0], signal[:-1]])
    turnover = np.abs(signal - position_prev)
    strat_log_returns = signal * log_returns - turnover * cost_pct
    cum_curve = np.exp(np.cumsum(strat_log_returns))
    return cum_curve


def fig08_trading_simulation(last_fold):
    seq_len = CONFIG["sequence_length"]
    seed = CONFIG["seed"]
    feature_sets = {
        "numeric_only": FEATURE_COLS_TECH,
        "numeric_plus_sentiment": FEATURE_COLS_TECH + FEATURE_COLS_SENT,
    }
    signals, idx_te_ref = {}, None
    for name, cols in feature_sets.items():
        scaler, scaled = fit_scaler_and_transform(last_fold["train"], dataset, cols)
        X_tr, yreg_tr, yclf_tr, _ = build_sequences(scaled, last_fold["train"], seq_len, dataset)
        X_val, yreg_val, yclf_val, _ = build_sequences(scaled, last_fold["val"], seq_len, dataset)
        X_te, yreg_te, yclf_te, idx_te = build_sequences(scaled, last_fold["test"], seq_len, dataset)
        model = train_lstm(X_tr, yreg_tr, yclf_tr, X_val, yreg_val, yclf_val, len(cols), seed)
        with torch.no_grad():
            _, pred_clf_logit = model(torch.tensor(X_te))
        pred_prob = torch.sigmoid(pred_clf_logit).numpy()
        signals[name] = np.where(pred_prob > 0.5, 1.0, -1.0)
        idx_te_ref = idx_te
        print(f"  fig08: modelo {name} treinado, n_test={len(idx_te)}")

    actual_log_returns = dataset.loc[idx_te_ref, "target_log_return"].values
    cost_pct = CONFIG["transaction_cost_pct"]

    cum_numeric = simulate_strategy(signals["numeric_only"], actual_log_returns, cost_pct)
    cum_sentiment = simulate_strategy(signals["numeric_plus_sentiment"], actual_log_returns, cost_pct)
    cum_buyhold = np.exp(np.cumsum(actual_log_returns))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(idx_te_ref, cum_numeric, color=PALETTE["primary"], label="Numérico puro")
    ax.plot(idx_te_ref, cum_sentiment, color=PALETTE["sentiment"], label="Numérico + sentimento")
    ax.plot(idx_te_ref, cum_buyhold, color=PALETTE["neutral"], linestyle="--", label="Buy-and-hold")
    ax.set_title("Retorno acumulado — simulação de trading")
    ax.set_ylabel("Capital (base = 1.0)")
    ax.text(0.01, 0.02, f"Custo de transação: {cost_pct * 100:.1f}% por operação — fold de teste {last_fold['fold']}",
            transform=ax.transAxes, fontsize=8, color="#555555")
    ax.legend()
    plt.tight_layout()
    path = FIGS_DIR / "08_trading_simulation.png"
    plt.savefig(path)
    plt.close(fig)
    print("Salvo:", path)
    print(f"  retorno final — numérico puro: {(cum_numeric[-1] - 1) * 100:.1f}% | "
          f"numérico+sentimento: {(cum_sentiment[-1] - 1) * 100:.1f}% | "
          f"buy-and-hold: {(cum_buyhold[-1] - 1) * 100:.1f}%")


if __name__ == "__main__":
    print("Carregando notícias + sentimento (cache)...")
    news = load_news_with_sentiment()

    fig01_price_overview()
    fig02_news_coverage(news)
    fig03_sentiment_distribution(news)
    fig03b_seasonality()
    fig04_pca_variance()
    fig05_correlation_heatmap()

    folds = walk_forward_splits(dataset.index, CONFIG["n_walk_forward_folds"], CONFIG["min_train_days"])
    fig06_walk_forward_scheme(folds)
    fig07_permutation_importance()
    fig08_trading_simulation(folds[-1])

    print("\nConcluído — figuras 01 a 08 regeneradas no estilo paper.")
