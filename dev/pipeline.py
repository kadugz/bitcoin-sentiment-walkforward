# %% [markdown]
# # Previsão de Movimento de Preços de Bitcoin: Dados Numéricos + Sentimento de Notícias
#
# **Trabalho final de IA.** Testamos se sentimento de notícias (FinBERT) agrega poder
# preditivo a um LSTM que já usa preço/volume/indicadores técnicos do BTC-USD, em regressão
# (fechamento t+1) e classificação direcional. Metodologia: baselines honestos, split
# walk-forward, múltiplas seeds com teste estatístico, interpretabilidade e simulação de
# trading. Discussão final à luz da Hipótese de Mercado Eficiente (EMH).

# %% [markdown]
# ## 1. Introdução
#
# **Pergunta de pesquisa:** sentimento de notícias (FinBERT) melhora, de forma
# estatisticamente significativa, a previsão do fechamento em t+1 de um LSTM que já usa
# preço/volume/indicadores técnicos?
#
# 1. **Regressão** do retorno log de fechamento em t+1 — RMSE, MAE, MAPE.
# 2. **Classificação direcional** (alta/queda) — acurácia, F1, AUC.
#
# **H0 (EMH semi-forte):** notícias já publicadas já estão precificadas, então sentimento
# agregado não deve ter poder preditivo incremental relevante. Um resultado nulo aqui é
# esperado e cientificamente informativo.

# %% [markdown]
# ## 0. Configuração Global

# %%
import os
import sys
import json
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Evita conflito de runtime OpenMP (numpy/MKL + torch) no Windows; precisa vir antes do
# primeiro `import torch`.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGS_DIR = PROJECT_ROOT / "figs"

_MNT_OUTPUTS = Path("/mnt/user-data/outputs")
OUTPUTS_DIR = _MNT_OUTPUTS if _MNT_OUTPUTS.parent.exists() else PROJECT_ROOT / "outputs"

for d in [DATA_RAW, DATA_PROCESSED, FIGS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "ticker": "BTC-USD",
    "price_start_date": "2020-01-01",
    "price_end_date": None,

    "horizon_days": 1,  # t+1

    "news_csv_path": str(DATA_RAW / "cryptonews.csv"),

    "finbert_model_name": "ProsusAI/finbert",
    "sentiment_text_field": "title",
    "sentiment_batch_size": 64,

    "sequence_length": 30,
    "seeds": [0, 1, 2, 3, 4],

    "n_walk_forward_folds": 5,
    "min_train_days": 365,

    # Mesmos hiperparâmetros para numérico puro e numérico+sentimento (isola o efeito das features)
    "lstm_hidden_size": 32,
    "lstm_num_layers": 1,
    "lstm_dropout": 0.0,
    "lstm_lr": 1e-3,
    "lstm_batch_size": 32,
    "lstm_max_epochs": 100,
    "lstm_patience": 10,

    "transaction_cost_pct": 0.001,

    "project_root": str(PROJECT_ROOT),
}

print(json.dumps({k: v for k, v in CONFIG.items() if k != "project_root"}, indent=2))


# %%
def set_global_seed(seed: int) -> None:
    """Fixa a seed em Python/NumPy/PyTorch."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)  # LSTM em CPU: modo determinístico é instável/lento


set_global_seed(CONFIG["seeds"][0])
print("Seed global inicial fixada:", CONFIG["seeds"][0])

# %% [markdown]
# ## 2. Coleta de Dados
#
# ### 2.1 Preço (yfinance)

# %%
import pandas as pd
import numpy as np

PRICE_CACHE = DATA_RAW / "btc_usd_ohlcv.csv"


def load_price_data(force_download: bool = False) -> pd.DataFrame:
    """Baixa (ou lê do cache) o OHLCV diário do BTC-USD via yfinance."""
    if PRICE_CACHE.exists() and not force_download:
        df = pd.read_csv(PRICE_CACHE, index_col=0, parse_dates=True)
        return df

    import yfinance as yf

    ticker = CONFIG["ticker"]
    df = yf.download(
        ticker,
        start=CONFIG["price_start_date"],
        end=CONFIG["price_end_date"],
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError(
            f"Falha ao baixar dados de {ticker} via yfinance. Verifique conexão com a "
            "internet. Não vou simular dados sintéticos — pare e me avise se isso ocorrer."
        )
    if isinstance(df.columns, pd.MultiIndex):  # yfinance recente pode devolver colunas MultiIndex
        df.columns = [c[0] for c in df.columns]
    df.index.name = "Date"
    df.to_csv(PRICE_CACHE)
    return df


price_df = load_price_data()
print("price_df shape:", price_df.shape)
print("price_df date range:", price_df.index.min(), "->", price_df.index.max())
print(price_df.head(3))
print(price_df.tail(3))

assert price_df.shape[0] > 3 * 365, "Menos de 3 anos de dados de preço — verifique a coleta."
expected_cols = {"Open", "High", "Low", "Close", "Volume"}
assert expected_cols.issubset(set(price_df.columns)), f"Colunas ausentes: {expected_cols - set(price_df.columns)}"
n_nan = price_df[list(expected_cols)].isna().sum().sum()
print("NaNs nas colunas OHLCV:", n_nan)
assert n_nan == 0, "Há NaNs inesperados nos dados de preço."

# %% [markdown]
# ### 2.2 Notícias de criptomoedas (Kaggle "Cryptocurrency News")
#
# `cryptonews.csv` (`oliviervha/crypto-news`). Se não estiver em `data/raw/`, tenta baixar
# via `kagglehub`; falhando, para com instrução de onde colocar o CSV manualmente. O dataset
# já vem com um campo `sentiment` pré-computado (aparentemente TextBlob) que **não usamos**
# como feature principal — a especificação pede FinBERT, mais adequado a texto financeiro.

# %%
NEWS_CSV_PATH = Path(CONFIG["news_csv_path"])
UPLOADS_DIR = Path("/mnt/user-data/uploads")


def find_local_news_csv() -> Path | None:
    """Procura o CSV de notícias em locais plausíveis; None se não achar (tenta kagglehub)."""
    candidates = [NEWS_CSV_PATH, UPLOADS_DIR / "cryptonews.csv"]
    if UPLOADS_DIR.exists():
        candidates += sorted(UPLOADS_DIR.glob("*.csv"))
    for c in candidates:
        if c.exists():
            return c
    return None


def load_news_raw() -> pd.DataFrame:
    local_path = find_local_news_csv()
    if local_path is not None:
        print(f"CSV de notícias encontrado localmente em: {local_path}")
        return pd.read_csv(local_path)

    print("CSV de notícias não encontrado localmente. Tentando baixar via kagglehub...")
    try:
        import kagglehub

        dataset_path = kagglehub.dataset_download("oliviervha/crypto-news")
        candidate = list(Path(dataset_path).glob("*.csv"))
        if not candidate:
            raise FileNotFoundError("kagglehub baixou o dataset mas nenhum CSV foi encontrado.")
        df = pd.read_csv(candidate[0])
        NEWS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(NEWS_CSV_PATH, index=False)
        return df
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Não foi possível obter o dataset de notícias automaticamente "
            f"(erro: {exc}).\n"
            "AÇÃO NECESSÁRIA: baixe manualmente o dataset 'Cryptocurrency News' "
            "(https://www.kaggle.com/datasets/oliviervha/crypto-news) e coloque o CSV em UM "
            f"destes locais:\n  - {NEWS_CSV_PATH}\n  - {UPLOADS_DIR / 'cryptonews.csv'} "
            "Se preferir usar kagglehub, configure a autenticação do Kaggle "
            "(kaggle.json em ~/.kaggle/ ou variáveis KAGGLE_USERNAME/KAGGLE_KEY)."
        ) from exc


news_raw = load_news_raw()
print("news_raw shape:", news_raw.shape)
print(news_raw.columns.tolist())
print(news_raw.head(2))

# %% [markdown]
# ### 2.3 Alinhamento temporal notícia → preço
#
# Timestamps das notícias em UTC; yfinance também fecha o dia à meia-noite UTC, então basta
# agregar por data-calendário. Sentimento agregado do dia `D` prevê o fechamento de `D+1`
# (`target_date = news_date + 1`) — a informação usada na previsão já é conhecida no momento
# em que ela seria feita, evitando look-ahead bias. Dias sem notícia recebem sentimento
# neutro imputado (nunca descartamos a linha de preço), com indicador binário `has_news`.

# %%
news = news_raw.copy()
news["date"] = pd.to_datetime(news["date"], errors="coerce", utc=True)
n_bad_dates = news["date"].isna().sum()
print(f"Linhas com data inválida/ausente (descartadas): {n_bad_dates} de {len(news)}")
news = news.dropna(subset=["date"]).copy()

news["news_date"] = news["date"].dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None)
news["target_date"] = news["news_date"] + pd.Timedelta(days=1)

print("Intervalo de datas das notícias (news_date):", news["news_date"].min(), "->", news["news_date"].max())
print("Notícias por assunto (subject):")
print(news["subject"].value_counts())

text_field = CONFIG["sentiment_text_field"]
n_empty_text = news[text_field].isna().sum()
print(f"Linhas com '{text_field}' vazio/NaN: {n_empty_text}")
news = news.dropna(subset=[text_field]).copy()
news[text_field] = news[text_field].astype(str).str.strip()
news = news[news[text_field].str.len() > 0].copy()
print("news shape após limpeza:", news.shape)

# %% [markdown]
# ### 2.4 Sentimento com FinBERT (ProsusAI/finbert)
#
# FinBERT em vez de VADER (léxico genérico, mal calibrado para linguagem financeira). Para
# cada headline: `sent_score = P(positivo) - P(negativo)` em [-1, 1] e a classe mais provável.
# Resultado cacheado em CSV (~31 mil headlines, CPU-only).

# %%
SENTIMENT_CACHE = DATA_PROCESSED / "news_finbert_sentiment.csv"


def run_finbert_sentiment(texts: list, model_name: str, batch_size: int = 64) -> pd.DataFrame:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    print("Mapeamento de classes do FinBERT:", id2label)

    all_probs = []
    n = len(texts)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = texts[start:start + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            if (start // batch_size) % 20 == 0:
                print(f"  FinBERT: {min(start + batch_size, n)}/{n} textos processados")

    probs = np.concatenate(all_probs, axis=0)
    cols = [id2label[i] for i in range(probs.shape[1])]
    probs_df = pd.DataFrame(probs, columns=cols)
    probs_df["sent_score"] = probs_df.get("positive", 0) - probs_df.get("negative", 0)
    probs_df["sent_class"] = probs_df[cols].idxmax(axis=1)
    return probs_df


if SENTIMENT_CACHE.exists():
    sent_df = pd.read_csv(SENTIMENT_CACHE)
    print("Sentimento FinBERT carregado do cache:", sent_df.shape)
    assert len(sent_df) == len(news), (
        "Cache de sentimento com tamanho diferente do dataset de notícias atual — "
        "apague data/processed/news_finbert_sentiment.csv e reexecute."
    )
else:
    print(f"Rodando FinBERT ({CONFIG['finbert_model_name']}) sobre {len(news)} headlines "
          "(primeira execução baixa o modelo do HuggingFace Hub; pode levar alguns minutos)...")
    sent_df = run_finbert_sentiment(
        news[text_field].tolist(),
        CONFIG["finbert_model_name"],
        batch_size=CONFIG["sentiment_batch_size"],
    )
    sent_df.to_csv(SENTIMENT_CACHE, index=False)
    print("Sentimento FinBERT salvo em cache:", SENTIMENT_CACHE)

news = news.reset_index(drop=True)
sent_df = sent_df.reset_index(drop=True)
news = pd.concat([news, sent_df], axis=1)

print(news[["news_date", text_field, "sent_score", "sent_class"]].head(5))
print("\nDistribuição de classes de sentimento (FinBERT):")
print(news["sent_class"].value_counts(normalize=True).round(3))

assert news["sent_score"].between(-1.0, 1.0).all(), "sent_score fora do intervalo esperado [-1, 1]."
assert news["sent_score"].isna().sum() == 0, "Há sent_score nulo — verifique a inferência do FinBERT."

# %% [markdown]
# ## 3. Análise Exploratória (Preço)

# %%
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
axes[0].plot(price_df.index, price_df["Close"], color="#1f77b4", linewidth=1)
axes[0].set_title(f"{CONFIG['ticker']} — Preço de fechamento diário")
axes[0].set_ylabel("Preço (USD)")
axes[0].grid(alpha=0.3)

axes[1].bar(price_df.index, price_df["Volume"], color="#7f7f7f", width=1.0)
axes[1].set_title("Volume diário")
axes[1].set_ylabel("Volume")
axes[1].grid(alpha=0.3)

plt.tight_layout()
fig_path = FIGS_DIR / "01_price_overview.png"
plt.savefig(fig_path, dpi=110)
plt.close(fig)
print("Figura salva em:", fig_path)

# %% [markdown]
# ### 3.1 Distribuição temporal das notícias
#
# O dataset Kaggle cobre só um subconjunto do período de preço (~out/2021 a dez/2023); fora
# disso o sentimento é imputado como neutro. Limitação discutida na Seção 12.

# %%
news_per_day = news.groupby("news_date").size()
full_range = pd.date_range(price_df.index.min(), price_df.index.max(), freq="D")
news_per_day_full = news_per_day.reindex(full_range, fill_value=0)

fig, ax = plt.subplots(figsize=(12, 4))
ax.fill_between(news_per_day_full.index, news_per_day_full.values, color="#2ca02c", alpha=0.6)
ax.axvspan(price_df.index.min(), news_per_day.index.min(), color="red", alpha=0.08, label="sem cobertura de notícias")
ax.axvspan(news_per_day.index.max(), price_df.index.max(), color="red", alpha=0.08)
ax.set_title("Nº de notícias por dia (cobertura do dataset Kaggle) vs. histórico de preço")
ax.set_ylabel("Notícias/dia")
ax.legend(loc="upper right")
plt.tight_layout()
fig_path2 = FIGS_DIR / "02_news_coverage.png"
plt.savefig(fig_path2, dpi=110)
plt.close(fig)
print("Figura salva em:", fig_path2)

pct_days_with_news_in_range = (news_per_day.index.max() - news_per_day.index.min()).days
total_days = (price_df.index.max() - price_df.index.min()).days
print(f"Dias com dataset de notícias disponível: {pct_days_with_news_in_range} de {total_days} "
      f"dias totais de preço ({100*pct_days_with_news_in_range/total_days:.1f}%)")

# %% [markdown]
# ### 3.2 Distribuição do sentimento (FinBERT)

# %%
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(news["sent_score"], bins=50, color="#9467bd", ax=axes[0])
axes[0].set_title("Distribuição de sent_score (P(pos) − P(neg))")
axes[0].set_xlabel("sent_score")

news["sent_class"].value_counts().reindex(["positive", "neutral", "negative"]).plot(
    kind="bar", color=["#2ca02c", "#7f7f7f", "#d62728"], ax=axes[1]
)
axes[1].set_title("Contagem por classe de sentimento")
axes[1].tick_params(axis="x", rotation=0)

plt.tight_layout()
fig_path3 = FIGS_DIR / "03_sentiment_distribution.png"
plt.savefig(fig_path3, dpi=110)
plt.close(fig)
print("Figura salva em:", fig_path3)
print(news["sent_score"].describe())

# %% [markdown]
# ### 3.3 Sazonalidade em múltiplas escalas (1, 30 e 365 dias)
#
# Decompomos o log-preço (tendência + sazonalidade + resíduo) em três periodicidades:
# `period=1` é degenerado por construção (checagem de sanidade, força deve sair 0), `period=30`
# testa padrões mensais, `period=365` testa padrões anuais/ciclo de halving. Resumimos cada
# escala pela força sazonal de Hyndman: `F_s = max(0, 1 - Var(resíduo)/Var(resíduo+sazonalidade))`.
# Com ~6-7 anos de histórico, `period=365` tem poucas repetições — força alta ali é indício
# exploratório, não sazonalidade de calendário estabelecida.

# %%
from statsmodels.tsa.seasonal import seasonal_decompose

SEASONAL_PERIODS = [1, 30, 365]

log_close_seasonal = np.log(price_df["Close"]).asfreq("D")
assert log_close_seasonal.isna().sum() == 0, "Gaps na série diária de preço — verifique a coleta."

seasonal_decomp_results = {}
seasonal_strength_rows = []
for period in SEASONAL_PERIODS:
    result = seasonal_decompose(
        log_close_seasonal, model="additive", period=period, extrapolate_trend="freq"
    )
    resid_var = np.nanvar(result.resid)
    seasonal_plus_resid_var = np.nanvar(result.resid + result.seasonal)
    # period=1: sazonalidade == resíduo -> divisão 0/0; força = 0 por definição.
    strength = 0.0 if seasonal_plus_resid_var == 0 else max(0.0, 1 - resid_var / seasonal_plus_resid_var)
    seasonal_decomp_results[period] = result
    seasonal_strength_rows.append({
        "period_days": period,
        "seasonal_strength": strength,
        "seasonal_amplitude": float(result.seasonal.max() - result.seasonal.min()),
    })

seasonal_strength_df = pd.DataFrame(seasonal_strength_rows)
print("Força sazonal (Hyndman) por escala testada:")
print(seasonal_strength_df.round(4).to_string(index=False))
seasonal_strength_df.to_csv(OUTPUTS_DIR / "seasonality_strength_by_period.csv", index=False)

fig, axes = plt.subplots(len(SEASONAL_PERIODS), 3, figsize=(14, 8), sharex=True)
for i, period in enumerate(SEASONAL_PERIODS):
    result = seasonal_decomp_results[period]
    strength = seasonal_strength_df.loc[seasonal_strength_df["period_days"] == period, "seasonal_strength"].iloc[0]
    axes[i, 0].plot(result.trend, color="#1f77b4", linewidth=1)
    axes[i, 0].set_ylabel(f"period={period}d\n\nTendência")
    axes[i, 0].grid(alpha=0.3)
    axes[i, 1].plot(result.seasonal, color="#2ca02c", linewidth=0.8)
    axes[i, 1].set_ylabel("Sazonalidade")
    axes[i, 1].set_title(f"força sazonal = {strength:.3f}", fontsize=9)
    axes[i, 1].grid(alpha=0.3)
    axes[i, 2].plot(result.resid, color="#d62728", linewidth=0.6)
    axes[i, 2].set_ylabel("Resíduo")
    axes[i, 2].grid(alpha=0.3)
axes[0, 0].set_title("Tendência (log-preço)")
axes[0, 1].set_title("Sazonalidade")
axes[0, 2].set_title("Resíduo")
fig.suptitle(f"{CONFIG['ticker']} — decomposição sazonal (log-preço) em 3 escalas", y=1.02)
plt.tight_layout()
fig_path3b = FIGS_DIR / "03b_seasonality_multiscale.png"
plt.savefig(fig_path3b, dpi=110, bbox_inches="tight")
plt.close(fig)
print("Figura salva em:", fig_path3b)

strongest_period = int(seasonal_strength_df.loc[seasonal_strength_df["seasonal_strength"].idxmax(), "period_days"])
print(f"\nEscala com maior força sazonal nesta execução: period={strongest_period} dias "
      f"(ver ressalva estatística no texto acima antes de tirar conclusões fortes).")

# %% [markdown]
# ## 4. Pré-processamento e Engenharia de Features
#
# ### 4.1 Indicadores técnicos de preço
#
# Retornos log (1/3/7d), médias móveis (7/14/30d) e razão preço/MA, volatilidade (7/14d),
# RSI-14 (Wilder), variação de volume — todos usando só informação até o dia `t`, então não
# há vazamento em relação ao alvo (`t + horizon_days`).

# %%
def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["Close"]
    volume = df["Volume"]

    log_ret_1 = np.log(close / close.shift(1))
    out["log_return_1d"] = log_ret_1
    out["log_return_3d"] = np.log(close / close.shift(3))
    out["log_return_7d"] = np.log(close / close.shift(7))

    for w in [7, 14, 30]:
        ma = close.rolling(w).mean()
        out[f"ma_{w}"] = ma
        out[f"price_to_ma_{w}"] = close / ma - 1.0

    for w in [7, 14]:
        out[f"volatility_{w}"] = log_ret_1.rolling(w).std()

    # RSI-14, suavização de Wilder via EWM
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    out["volume_change_1d"] = volume.pct_change(1)
    out["volume_ma_7"] = volume.rolling(7).mean()

    out["close"] = close  # só para montar o alvo; removido das features de entrada depois
    return out


tech_features = compute_technical_features(price_df)
print("tech_features shape (antes de descartar warm-up):", tech_features.shape)
print("NaNs por coluna (esperado apenas nas janelas iniciais de warm-up, ~30 primeiros dias):")
print(tech_features.isna().sum())

# %% [markdown]
# ### 4.2 Agregação diária de sentimento
#
# Por `news_date`: `sent_mean`/`sent_median`/`sent_std`, `news_count`, proporções de cada
# classe, e `has_news` (0 = dia sem notícia, sentimento imputado neutro).

# %%
def aggregate_daily_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    g = news_df.groupby("news_date")
    agg = pd.DataFrame(
        {
            "sent_mean": g["sent_score"].mean(),
            "sent_std": g["sent_score"].std(),
            "sent_median": g["sent_score"].median(),
            "news_count": g.size(),
        }
    )
    class_dummies = pd.get_dummies(news_df["sent_class"])
    props = class_dummies.groupby(news_df["news_date"]).mean()
    props.columns = [f"sent_prop_{c}" for c in props.columns]
    agg = agg.join(props, how="left")
    return agg


daily_sent = aggregate_daily_sentiment(news)
print("daily_sent shape (só dias com notícia):", daily_sent.shape)
print(daily_sent.head())

daily_sent_full = daily_sent.reindex(price_df.index)
daily_sent_full["has_news"] = daily_sent_full["news_count"].notna().astype(int)
fill_zero_cols = [c for c in daily_sent_full.columns if c != "has_news"]
daily_sent_full[fill_zero_cols] = daily_sent_full[fill_zero_cols].fillna(0.0)
daily_sent_full["news_count"] = daily_sent_full["news_count"].astype(int)
daily_sent_full["sent_std"] = daily_sent_full["sent_std"].fillna(0.0)  # dias com 1 notícia -> std NaN -> 0

assert daily_sent_full.isna().sum().sum() == 0, "Imputação neutra incompleta — há NaN remanescente."
print("\ndaily_sent_full shape (calendário completo do preço):", daily_sent_full.shape)
print("Dias com has_news=1:", int(daily_sent_full["has_news"].sum()), "de", len(daily_sent_full))

# %% [markdown]
# ### 4.3 Tabela unificada, alvo e janela de modelagem
#
# Notícias cobrem só ~33% do histórico de preço; incluir o resto diluiria o efeito do
# sentimento (ficaria majoritariamente neutro imputado). Por isso restringimos toda a
# modelagem ao intervalo `[min(news_date), max(news_date)]`.

# %%
feature_cols_tech = [c for c in tech_features.columns if c != "close"]
feature_cols_sent = [c for c in daily_sent_full.columns]

features_full = tech_features[feature_cols_tech].join(daily_sent_full, how="left")
features_full["close"] = tech_features["close"]

H = CONFIG["horizon_days"]
features_full["target_close"] = features_full["close"].shift(-H)
features_full["target_log_return"] = np.log(features_full["target_close"] / features_full["close"])
features_full["target_direction"] = (features_full["target_close"] > features_full["close"]).astype(int)

MODELING_START = daily_sent.index.min()
MODELING_END = daily_sent.index.max()
print(f"Janela de modelagem definida pela cobertura de notícias: {MODELING_START.date()} -> {MODELING_END.date()}")

dataset = features_full.loc[MODELING_START:MODELING_END].copy()
print("dataset shape (dentro da janela, antes de dropna de warm-up/alvo):", dataset.shape)

n_nan_before = dataset.isna().sum()
print("\nNaNs por coluna dentro da janela de modelagem:")
print(n_nan_before[n_nan_before > 0])

dataset = dataset.dropna()
print("\ndataset shape final (após dropna):", dataset.shape)
assert dataset.isna().sum().sum() == 0

# Checagem numérica anti-look-ahead: alvo(t) deve ser exatamente Close(t+H)
check = dataset.index + pd.Timedelta(days=H)
implied_target_close = price_df["Close"].reindex(check).values
assert np.allclose(dataset["target_close"].values, implied_target_close, equal_nan=False), (
    "Alvo não corresponde ao fechamento de t+H — possível bug de alinhamento/look-ahead."
)
print(f"\nChecagem de alinhamento OK: target_close(t) == Close(t + {H} dia(s)) para todas as {len(dataset)} linhas.")

FEATURE_COLUMNS = feature_cols_tech + feature_cols_sent
print(f"\nTotal de features de entrada: {len(FEATURE_COLUMNS)}")
print(FEATURE_COLUMNS)

dataset.to_csv(DATA_PROCESSED / "dataset_features_target.csv")

# %% [markdown]
# ### 4.4 PCA exploratório
#
# Só diagnóstico (redundância entre indicadores, variância que o sentimento agrega) — não
# entra no pipeline de modelagem, que usa as features originais.

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_pca_input = dataset[FEATURE_COLUMNS].copy()
scaler_pca = StandardScaler().fit(X_pca_input)
X_pca_scaled = scaler_pca.transform(X_pca_input)

pca = PCA(n_components=min(15, X_pca_scaled.shape[1]))
pca.fit(X_pca_scaled)
explained = pca.explained_variance_ratio_
cum_explained = np.cumsum(explained)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(range(1, len(explained) + 1), explained, color="#1f77b4")
axes[0].set_title("Variância explicada por componente")
axes[0].set_xlabel("Componente")
axes[0].set_ylabel("Proporção da variância")

axes[1].plot(range(1, len(cum_explained) + 1), cum_explained, marker="o", color="#ff7f0e")
axes[1].axhline(0.9, color="gray", linestyle="--", linewidth=1)
axes[1].set_title("Variância explicada acumulada")
axes[1].set_xlabel("Nº de componentes")
axes[1].set_ylabel("Acumulada")

plt.tight_layout()
fig_path4 = FIGS_DIR / "04_pca_variance.png"
plt.savefig(fig_path4, dpi=110)
plt.close(fig)
print("Figura salva em:", fig_path4)

n_components_90 = int(np.searchsorted(cum_explained, 0.9) + 1)
print(f"Nº de componentes para explicar 90% da variância: {n_components_90} de {X_pca_scaled.shape[1]} features originais")

loadings = pd.DataFrame(pca.components_[:2].T, index=FEATURE_COLUMNS, columns=["PC1", "PC2"])
print("\nLoadings das features de sentimento nos 2 primeiros PCs:")
print(loadings.loc[[c for c in FEATURE_COLUMNS if c in feature_cols_sent]])

# %% [markdown]
# ### 4.5 Correlações (preço, sentimento, alvo)

# %%
corr_cols = [
    "log_return_1d", "volatility_7", "rsi_14", "price_to_ma_30",
    "sent_mean", "sent_prop_positive", "sent_prop_negative", "news_count", "has_news",
    "target_log_return",
]
corr_matrix = dataset[corr_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, vmin=-1, vmax=1)
ax.set_title("Correlação entre features-chave e o alvo (retorno log t+1)")
plt.tight_layout()
fig_path5 = FIGS_DIR / "05_correlation_heatmap.png"
plt.savefig(fig_path5, dpi=110)
plt.close(fig)
print("Figura salva em:", fig_path5)
print("\nCorrelação de cada feature de sentimento com o alvo:")
sent_target_corr = dataset[feature_cols_sent + ["target_log_return"]].corr()["target_log_return"]
print(sent_target_corr[feature_cols_sent].sort_values())

# %% [markdown]
# ## 5. Split Temporal Walk-Forward
#
# Nunca split aleatório. Janela expandindo: treino cresce a cada fold, val/test sempre à
# frente no tempo. `StandardScaler` ajustado só no treino de cada fold.

# %%
def walk_forward_splits(index: pd.DatetimeIndex, n_folds: int, min_train_days: int, val_frac: float = 0.15):
    """Folds walk-forward (expanding window): train < val < test, cronológico, sem overlap."""
    n = len(index)
    assert n > min_train_days, "Dataset menor que o tamanho mínimo de treino configurado."
    remaining = n - min_train_days
    fold_size = remaining // n_folds
    assert fold_size > 10, "Poucos dias por fold — reduza n_folds ou min_train_days."

    folds = []
    for k in range(n_folds):
        train_end = min_train_days + k * fold_size
        test_end = min_train_days + (k + 1) * fold_size if k < n_folds - 1 else n
        train_full_idx = index[:train_end]
        test_idx = index[train_end:test_end]
        n_val = max(5, int(len(train_full_idx) * val_frac))
        train_idx = train_full_idx[:-n_val]
        val_idx = train_full_idx[-n_val:]
        folds.append({"fold": k, "train": train_idx, "val": val_idx, "test": test_idx})
    return folds


WALK_FORWARD_FOLDS = walk_forward_splits(
    dataset.index, CONFIG["n_walk_forward_folds"], CONFIG["min_train_days"]
)

print(f"{len(WALK_FORWARD_FOLDS)} folds walk-forward gerados:\n")
for f in WALK_FORWARD_FOLDS:
    print(
        f"  Fold {f['fold']}: train[{f['train'].min().date()} -> {f['train'].max().date()}] "
        f"({len(f['train'])}d)  val[{f['val'].min().date()} -> {f['val'].max().date()}] "
        f"({len(f['val'])}d)  test[{f['test'].min().date()} -> {f['test'].max().date()}] "
        f"({len(f['test'])}d)"
    )

for f in WALK_FORWARD_FOLDS:
    assert f["train"].max() < f["val"].min(), "Treino se sobrepõe/ultrapassa validação."
    assert f["val"].max() < f["test"].min(), "Validação se sobrepõe/ultrapassa teste."
    assert len(set(f["train"]) & set(f["val"]) & set(f["test"])) == 0, "Overlap entre partições do fold."
print("\nValidação estrutural OK: train < val < test (estritamente cronológico) em todos os folds, sem overlap.")

fig, ax = plt.subplots(figsize=(11, 3.2))
colors = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}
for f in WALK_FORWARD_FOLDS:
    y = f["fold"]
    for part in ["train", "val", "test"]:
        idx = f[part]
        ax.barh(y, (idx.max() - idx.min()).days + 1, left=idx.min(), height=0.6, color=colors[part], label=part if y == 0 else None)
ax.set_yticks([f["fold"] for f in WALK_FORWARD_FOLDS])
ax.set_yticklabels([f"Fold {f['fold']}" for f in WALK_FORWARD_FOLDS])
ax.set_title("Esquema walk-forward (expanding window) — train/val/test por fold")
ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()
fig_path6 = FIGS_DIR / "06_walk_forward_scheme.png"
plt.savefig(fig_path6, dpi=110)
plt.close(fig)
print("Figura salva em:", fig_path6)

# %% [markdown]
# ## 6. Baselines Honestos
#
# Persistência ingênua (`close(t+H)=close(t)`; direção repete a última observada) e ARIMA
# univariado sobre `log(close)` (ordem por AIC, atualizado dia a dia via `refit=False`).

# %%
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,
    accuracy_score, f1_score, confusion_matrix, roc_auc_score,
)


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def classification_metrics(y_true, y_pred, y_score=None) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out["confusion_matrix"] = cm.tolist()
    if y_score is not None and len(np.unique(y_true)) > 1:
        out["auc"] = float(roc_auc_score(y_true, y_score))
    else:
        out["auc"] = float("nan")
    return out


# %% [markdown]
# ### 6.1 Persistência ingênua

# %%
def naive_persistence_predictions(fold: dict) -> dict:
    idx = fold["test"]
    close_t = dataset.loc[idx, "close"]
    target_close = dataset.loc[idx, "target_close"]
    pred_close = close_t.copy()  # regressão: prevê close(t+H) = close(t)

    prev_direction = (dataset.loc[idx, "log_return_1d"] > 0).astype(int)  # direção t-1 -> t
    true_direction = dataset.loc[idx, "target_direction"]

    reg_m = regression_metrics(target_close, pred_close)
    clf_m = classification_metrics(true_direction, prev_direction, y_score=prev_direction.astype(float))
    return {"regression": reg_m, "classification": clf_m}


baseline_naive_results = [
    {"fold": f["fold"], **naive_persistence_predictions(f)} for f in WALK_FORWARD_FOLDS
]
for r in baseline_naive_results:
    print(f"Fold {r['fold']} | Persistência | RMSE={r['regression']['RMSE']:.2f} "
          f"MAE={r['regression']['MAE']:.2f} MAPE={r['regression']['MAPE']:.4f} | "
          f"Acc={r['classification']['accuracy']:.3f} F1={r['classification']['f1']:.3f}")

# %% [markdown]
# ### 6.2 ARIMA univariado

# %%
from statsmodels.tsa.arima.model import ARIMA
import itertools

log_close_full = np.log(price_df["Close"]).asfreq("D")
assert log_close_full.isna().sum() == 0, "Gaps na série diária de preço — reindexação de frequência introduziu NaN."


def select_arima_order(train_series: pd.Series, p_range=range(0, 3), q_range=range(0, 3), d=1):
    best_aic, best_order = np.inf, (1, d, 0)
    for p, q in itertools.product(p_range, q_range):
        if p == 0 and q == 0:
            continue
        try:
            fit = ARIMA(train_series, order=(p, d, q), enforce_stationarity=False, enforce_invertibility=False).fit()
            if fit.aic < best_aic:
                best_aic, best_order = fit.aic, (p, d, q)
        except Exception:
            continue
    return best_order, best_aic


def rolling_arima_forecast(fold: dict, horizon: int) -> pd.Series:
    train_end = fold["train"].max()
    history = log_close_full.loc[:train_end]
    order, aic = select_arima_order(history)
    model = ARIMA(history, order=order, enforce_stationarity=False, enforce_invertibility=False).fit()
    print(f"    ARIMA ordem selecionada (fold {fold['fold']}): {order} (AIC={aic:.1f})")

    eval_idx = fold["val"].append(fold["test"]).sort_values()
    preds = {}
    for t in eval_idx:
        model = model.append(log_close_full.loc[[t]], refit=False)
        fc = model.forecast(steps=horizon)
        preds[t] = fc.iloc[-1]
    preds = pd.Series(preds)
    return np.exp(preds.loc[fold["test"]])


baseline_arima_results = []
for f in WALK_FORWARD_FOLDS:
    print(f"  Ajustando ARIMA para fold {f['fold']}...")
    pred_close_arima = rolling_arima_forecast(f, CONFIG["horizon_days"])
    true_close = dataset.loc[f["test"], "target_close"]

    pred_direction = (pred_close_arima.values > dataset.loc[f["test"], "close"].values).astype(int)
    true_direction = dataset.loc[f["test"], "target_direction"].values

    reg_m = regression_metrics(true_close.values, pred_close_arima.values)
    clf_m = classification_metrics(true_direction, pred_direction, y_score=pred_direction.astype(float))
    baseline_arima_results.append({"fold": f["fold"], "regression": reg_m, "classification": clf_m})
    print(f"Fold {f['fold']} | ARIMA | RMSE={reg_m['RMSE']:.2f} MAE={reg_m['MAE']:.2f} "
          f"MAPE={reg_m['MAPE']:.4f} | Acc={clf_m['accuracy']:.3f} F1={clf_m['f1']:.3f}")

# %% [markdown]
# ### 6.3 Resumo dos baselines (média entre folds)

# %%
def summarize_fold_results(results: list, model_name: str) -> dict:
    reg_keys = ["RMSE", "MAE", "MAPE"]
    clf_keys = ["accuracy", "f1", "auc"]
    summary = {"model": model_name}
    for k in reg_keys:
        vals = [r["regression"][k] for r in results]
        summary[f"reg_{k}_mean"] = float(np.mean(vals))
        summary[f"reg_{k}_std"] = float(np.std(vals))
    for k in clf_keys:
        vals = [r["classification"][k] for r in results if not np.isnan(r["classification"][k])]
        summary[f"clf_{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
        summary[f"clf_{k}_std"] = float(np.std(vals)) if vals else float("nan")
    return summary


baseline_summary = pd.DataFrame(
    [
        summarize_fold_results(baseline_naive_results, "Persistência ingênua"),
        summarize_fold_results(baseline_arima_results, "ARIMA"),
    ]
)
print(baseline_summary.round(4).to_string(index=False))
baseline_summary.to_csv(OUTPUTS_DIR / "baseline_results_summary.csv", index=False)

# %% [markdown]
# ## 7. Modelo Numérico Puro (LSTM)
#
# Arquitetura/hiperparâmetros/seeds idênticos para numérico puro (Seção 7) e
# numérico+sentimento (Seção 8) — só as features de entrada mudam, isolando o efeito do
# sentimento. Encoder LSTM compartilhado + duas cabeças (regressão + classificação),
# perda combinada MSE+BCE, 5 seeds por combinação fold×features (base do teste estatístico
# da Seção 9). Embaralhamos a ordem dos exemplos em cada época (SGD padrão), não o tempo
# dentro de cada janela.

# %%
import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.reg_head = nn.Linear(hidden_size, 1)
        self.clf_head = nn.Linear(hidden_size, 1)  # logit binário

    def forward(self, x):
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        reg_out = self.reg_head(last_step).squeeze(-1)
        clf_out = self.clf_head(last_step).squeeze(-1)
        return reg_out, clf_out


def fit_scaler_and_transform(fold: dict, feature_cols: list) -> pd.DataFrame:
    """Ajusta o StandardScaler só no treino do fold; aplica na série toda."""
    scaler = StandardScaler().fit(dataset.loc[fold["train"], feature_cols])
    scaled = pd.DataFrame(
        scaler.transform(dataset[feature_cols]), index=dataset.index, columns=feature_cols
    )
    return scaler, scaled


def build_sequences(scaled_features: pd.DataFrame, target_idx: pd.DatetimeIndex, seq_len: int):
    full_idx = scaled_features.index
    pos_map = {d: i for i, d in enumerate(full_idx)}
    X_list, y_reg_list, y_clf_list, kept_idx = [], [], [], []
    for d in target_idx:
        pos = pos_map[d]
        if pos - seq_len + 1 < 0:
            continue  # sem histórico suficiente (só perto do início do dataset)
        window = scaled_features.iloc[pos - seq_len + 1: pos + 1].values
        X_list.append(window)
        y_reg_list.append(dataset.loc[d, "target_log_return"])
        y_clf_list.append(dataset.loc[d, "target_direction"])
        kept_idx.append(d)
    X = np.stack(X_list).astype(np.float32)
    y_reg = np.array(y_reg_list, dtype=np.float32)
    y_clf = np.array(y_clf_list, dtype=np.float32)
    return X, y_reg, y_clf, pd.DatetimeIndex(kept_idx)


def train_lstm(X_train, y_reg_train, y_clf_train, X_val, y_reg_val, y_clf_val, n_features, seed):
    set_global_seed(seed)
    model = LSTMForecaster(
        n_features, CONFIG["lstm_hidden_size"], CONFIG["lstm_num_layers"], CONFIG["lstm_dropout"]
    )
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lstm_lr"])
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()

    X_train_t = torch.tensor(X_train)
    y_reg_train_t = torch.tensor(y_reg_train)
    y_clf_train_t = torch.tensor(y_clf_train)
    X_val_t = torch.tensor(X_val)
    y_reg_val_t = torch.tensor(y_reg_val)
    y_clf_val_t = torch.tensor(y_clf_val)

    best_val_loss, best_state, bad_epochs = np.inf, None, 0
    n = X_train_t.shape[0]
    bs = CONFIG["lstm_batch_size"]
    g = torch.Generator().manual_seed(seed)

    for epoch in range(CONFIG["lstm_max_epochs"]):
        model.train()
        perm = torch.randperm(n, generator=g)
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            xb, yb_reg, yb_clf = X_train_t[idx], y_reg_train_t[idx], y_clf_train_t[idx]
            opt.zero_grad()
            pred_reg, pred_clf = model(xb)
            loss = mse(pred_reg, yb_reg) + bce(pred_clf, yb_clf)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_reg, val_clf = model(X_val_t)
            val_loss = (mse(val_reg, y_reg_val_t) + bce(val_clf, y_clf_val_t)).item()
        if val_loss < best_val_loss - 1e-6:
            best_val_loss, bad_epochs = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= CONFIG["lstm_patience"]:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def evaluate_lstm(model, X_test, y_reg_test, y_clf_test, close_t, target_close_true):
    with torch.no_grad():
        pred_reg, pred_clf_logit = model(torch.tensor(X_test))
    pred_reg = pred_reg.numpy()
    pred_clf_prob = torch.sigmoid(pred_clf_logit).numpy()
    pred_clf_label = (pred_clf_prob > 0.5).astype(int)

    pred_close = close_t * np.exp(pred_reg)  # retorno log previsto -> preço
    reg_m = regression_metrics(target_close_true, pred_close)
    clf_m = classification_metrics(y_clf_test.astype(int), pred_clf_label, y_score=pred_clf_prob)
    return reg_m, clf_m, pred_reg, pred_clf_prob


def run_lstm_experiment(feature_cols: list, feature_set_name: str) -> pd.DataFrame:
    rows = []
    for fold in WALK_FORWARD_FOLDS:
        scaler, scaled = fit_scaler_and_transform(fold, feature_cols)
        X_tr, yreg_tr, yclf_tr, idx_tr = build_sequences(scaled, fold["train"], CONFIG["sequence_length"])
        X_val, yreg_val, yclf_val, idx_val = build_sequences(scaled, fold["val"], CONFIG["sequence_length"])
        X_te, yreg_te, yclf_te, idx_te = build_sequences(scaled, fold["test"], CONFIG["sequence_length"])

        close_t_test = dataset.loc[idx_te, "close"].values
        target_close_test = dataset.loc[idx_te, "target_close"].values

        for seed in CONFIG["seeds"]:
            model = train_lstm(X_tr, yreg_tr, yclf_tr, X_val, yreg_val, yclf_val, len(feature_cols), seed)
            reg_m, clf_m, _, _ = evaluate_lstm(model, X_te, yreg_te, yclf_te, close_t_test, target_close_test)
            rows.append({
                "feature_set": feature_set_name, "fold": fold["fold"], "seed": seed,
                **{f"reg_{k}": v for k, v in reg_m.items()},
                **{f"clf_{k}": v for k, v in clf_m.items() if k != "confusion_matrix"},
            })
        print(f"  [{feature_set_name}] fold {fold['fold']} concluído "
              f"({len(CONFIG['seeds'])} seeds, n_test={len(idx_te)})")
    return pd.DataFrame(rows)


NUMERIC_ONLY_FEATURES = feature_cols_tech
NUMERIC_SENTIMENT_FEATURES = feature_cols_tech + feature_cols_sent

print("### Treinando modelo NUMÉRICO PURO (Seção 7) ###")
results_numeric = run_lstm_experiment(NUMERIC_ONLY_FEATURES, "numeric_only")

# %% [markdown]
# ## 8. Modelo Numérico + Sentimento (LSTM)
#
# Mesma arquitetura/seeds da Seção 7, agora com features técnicas + sentimento.

# %%
print("### Treinando modelo NUMÉRICO + SENTIMENTO (Seção 8) ###")
results_sentiment = run_lstm_experiment(NUMERIC_SENTIMENT_FEATURES, "numeric_plus_sentiment")

lstm_results = pd.concat([results_numeric, results_sentiment], ignore_index=True)
lstm_results.to_csv(OUTPUTS_DIR / "lstm_results_raw.csv", index=False)
print("\nlstm_results shape:", lstm_results.shape)
print(lstm_results.groupby("feature_set")[["reg_RMSE", "reg_MAE", "clf_accuracy", "clf_f1", "clf_auc"]].agg(["mean", "std"]).round(4))

# %% [markdown]
# ## 9. Comparação de Métricas e Teste de Significância Estatística
#
# Pares (fold, seed) do numérico puro vs. numérico+sentimento — mesma seed/fold em ambos, o
# que torna a comparação pareada por construção.

# %%
def summarize_metric_df(df: pd.DataFrame, model_name: str) -> dict:
    summary = {"model": model_name}
    for k in ["RMSE", "MAE", "MAPE"]:
        summary[f"reg_{k}_mean"] = df[f"reg_{k}"].mean()
        summary[f"reg_{k}_std"] = df[f"reg_{k}"].std()
    for k in ["accuracy", "f1", "auc"]:
        summary[f"clf_{k}_mean"] = df[f"clf_{k}"].mean()
        summary[f"clf_{k}_std"] = df[f"clf_{k}"].std()
    return summary


final_comparison = pd.DataFrame(
    [
        *baseline_summary.to_dict("records"),
        summarize_metric_df(results_numeric, "LSTM numérico puro"),
        summarize_metric_df(results_sentiment, "LSTM numérico + sentimento"),
    ]
)
print("### Tabela final de comparação (média ± desvio-padrão entre folds/seeds) ###")
print(final_comparison.round(4).to_string(index=False))
final_comparison.to_csv(OUTPUTS_DIR / "final_comparison_table.csv", index=False)

# %% [markdown]
# ### 9.1 Teste estatístico pareado (t de Student e Wilcoxon)
#
# H0: a distribuição da métrica é igual entre numérico puro e numérico+sentimento. Rejeitamos
# se p < 0.05.

# %%
from scipy import stats

paired = results_numeric.merge(results_sentiment, on=["fold", "seed"], suffixes=("_numeric", "_sentiment"))
print(f"Nº de pares (fold, seed) para o teste: {len(paired)}")

metrics_to_test = ["reg_RMSE", "reg_MAE", "clf_accuracy", "clf_f1"]
test_rows = []
for m in metrics_to_test:
    a = paired[f"{m}_numeric"].values
    b = paired[f"{m}_sentiment"].values
    diff = b - a
    t_stat, t_p = stats.ttest_rel(b, a)
    try:
        w_stat, w_p = stats.wilcoxon(b, a)
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    test_rows.append({
        "metric": m,
        "mean_numeric": a.mean(),
        "mean_sentiment": b.mean(),
        "mean_diff_(sentimento_menos_numerico)": diff.mean(),
        "t_stat": t_stat, "t_pvalue": t_p,
        "wilcoxon_stat": w_stat, "wilcoxon_pvalue": w_p,
        "significativo_5pct": bool(t_p < 0.05),
    })

stat_test_df = pd.DataFrame(test_rows)
print(stat_test_df.round(4).to_string(index=False))
stat_test_df.to_csv(OUTPUTS_DIR / "statistical_significance_tests.csv", index=False)
print(
    "\nNota de leitura: para RMSE/MAE, diferença negativa = sentimento REDUZ o erro (melhora); "
    "para accuracy/F1, diferença positiva = sentimento AUMENTA a métrica (melhora)."
)

# %% [markdown]
# ## 10. Interpretabilidade
#
# Permutation importance como método principal (mais robusto/barato que SHAP genérico para
# uma arquitetura recorrente): embaralha uma feature por vez nos exemplos de teste e mede
# quanto o RMSE piora. Tentamos SHAP também como fallback best-effort (Seção 10.1).

# %%
def permutation_importance_lstm(model, X_test, y_reg_test, y_clf_test, close_t_test,
                                 target_close_test, feature_names, seed=0, n_repeats=5):
    rng = np.random.default_rng(seed)
    base_reg, base_clf, _, _ = evaluate_lstm(model, X_test, y_reg_test, y_clf_test, close_t_test, target_close_test)
    base_rmse, base_acc = base_reg["RMSE"], base_clf["accuracy"]

    rows = []
    n_samples = X_test.shape[0]
    for j, fname in enumerate(feature_names):
        rmse_deltas, acc_drops = [], []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            perm_idx = rng.permutation(n_samples)
            X_perm[:, :, j] = X_perm[perm_idx, :, j]
            reg_m, clf_m, _, _ = evaluate_lstm(model, X_perm, y_reg_test, y_clf_test, close_t_test, target_close_test)
            rmse_deltas.append(reg_m["RMSE"] - base_rmse)
            acc_drops.append(base_acc - clf_m["accuracy"])
        rows.append({
            "feature": fname,
            "delta_RMSE": float(np.mean(rmse_deltas)),
            "accuracy_drop": float(np.mean(acc_drops)),
            "is_sentiment_feature": fname in feature_cols_sent,
        })
    return pd.DataFrame(rows).sort_values("delta_RMSE", ascending=False)


# Retreina (seed fixa) o modelo numérico+sentimento no último fold, para interpretabilidade
last_fold = WALK_FORWARD_FOLDS[-1]
interp_seed = CONFIG["seeds"][0]
scaler_i, scaled_i = fit_scaler_and_transform(last_fold, NUMERIC_SENTIMENT_FEATURES)
X_tr_i, yreg_tr_i, yclf_tr_i, _ = build_sequences(scaled_i, last_fold["train"], CONFIG["sequence_length"])
X_val_i, yreg_val_i, yclf_val_i, _ = build_sequences(scaled_i, last_fold["val"], CONFIG["sequence_length"])
X_te_i, yreg_te_i, yclf_te_i, idx_te_i = build_sequences(scaled_i, last_fold["test"], CONFIG["sequence_length"])
close_t_te_i = dataset.loc[idx_te_i, "close"].values
target_close_te_i = dataset.loc[idx_te_i, "target_close"].values

model_interp = train_lstm(X_tr_i, yreg_tr_i, yclf_tr_i, X_val_i, yreg_val_i, yclf_val_i,
                           len(NUMERIC_SENTIMENT_FEATURES), interp_seed)

perm_importance_df = permutation_importance_lstm(
    model_interp, X_te_i, yreg_te_i, yclf_te_i, close_t_te_i, target_close_te_i,
    NUMERIC_SENTIMENT_FEATURES, seed=interp_seed,
)
print(perm_importance_df.round(5).to_string(index=False))
perm_importance_df.to_csv(OUTPUTS_DIR / "permutation_importance.csv", index=False)

sent_total_importance = perm_importance_df.loc[perm_importance_df["is_sentiment_feature"], "delta_RMSE"].sum()
tech_total_importance = perm_importance_df.loc[~perm_importance_df["is_sentiment_feature"], "delta_RMSE"].sum()
print(f"\nSoma de delta_RMSE — features de sentimento: {sent_total_importance:.5f} | "
      f"features técnicas: {tech_total_importance:.5f}")

fig, ax = plt.subplots(figsize=(9, 6))
plot_df = perm_importance_df.sort_values("delta_RMSE")
colors_imp = ["#d62728" if s else "#1f77b4" for s in plot_df["is_sentiment_feature"]]
ax.barh(plot_df["feature"], plot_df["delta_RMSE"], color=colors_imp)
ax.set_xlabel("Aumento no RMSE ao embaralhar a feature (importância)")
ax.set_title("Permutation importance — modelo numérico + sentimento (vermelho = sentimento)")
plt.tight_layout()
fig_path7 = FIGS_DIR / "07_permutation_importance.png"
plt.savefig(fig_path7, dpi=110)
plt.close(fig)
print("Figura salva em:", fig_path7)

# %% [markdown]
# ### 10.1 Tentativa de SHAP (best-effort)

# %%
try:
    import shap

    background = torch.tensor(X_tr_i[np.random.default_rng(0).choice(len(X_tr_i), size=min(32, len(X_tr_i)), replace=False)])
    explain_sample = torch.tensor(X_te_i[: min(32, len(X_te_i))])

    reg_only_model = lambda x: model_interp(x)[0].unsqueeze(-1)  # SHAP espera saída (n, 1)
    explainer = shap.GradientExplainer(reg_only_model, background)
    shap_values = explainer.shap_values(explain_sample)
    shap_arr = np.array(shap_values)
    if shap_arr.ndim == 4:
        shap_arr = shap_arr[..., 0]
    shap_feature_importance = np.abs(shap_arr).mean(axis=(0, 1))  # agrega |SHAP| pela dimensão temporal
    shap_df = pd.DataFrame({
        "feature": NUMERIC_SENTIMENT_FEATURES,
        "mean_abs_shap": shap_feature_importance,
        "is_sentiment_feature": [f in feature_cols_sent for f in NUMERIC_SENTIMENT_FEATURES],
    }).sort_values("mean_abs_shap", ascending=False)
    print("SHAP (GradientExplainer) executado com sucesso:")
    print(shap_df.round(5).to_string(index=False))
    shap_df.to_csv(OUTPUTS_DIR / "shap_importance.csv", index=False)
    SHAP_OK = True
except Exception as exc:  # noqa: BLE001
    print(f"SHAP não pôde ser executado nesta arquitetura sequencial (motivo: {exc}).")
    print("Seguindo apenas com permutation importance (fallback previsto na especificação do trabalho).")
    SHAP_OK = False

# %% [markdown]
# ## 11. Simulação de Trading com Custos de Transação
#
# No teste do último fold: posição `+1` se prevê alta, `-1` se prevê queda; retorno =
# `posição × retorno_log_real - custo`, custo de 0.1% por unidade de posição alterada (virar
# de +1 para -1 custa 0.2%, duas operações). Comparado a buy-and-hold. O custo aditivo sobre
# o retorno log é aproximação padrão para custos pequenos; não modela slippage/liquidez.

# %%
def train_for_trading(feature_cols, fold, seed):
    scaler, scaled = fit_scaler_and_transform(fold, feature_cols)
    X_tr, yreg_tr, yclf_tr, _ = build_sequences(scaled, fold["train"], CONFIG["sequence_length"])
    X_val, yreg_val, yclf_val, _ = build_sequences(scaled, fold["val"], CONFIG["sequence_length"])
    X_te, yreg_te, yclf_te, idx_te = build_sequences(scaled, fold["test"], CONFIG["sequence_length"])
    model = train_lstm(X_tr, yreg_tr, yclf_tr, X_val, yreg_val, yclf_val, len(feature_cols), seed)
    with torch.no_grad():
        _, pred_clf_logit = model(torch.tensor(X_te))
    pred_prob = torch.sigmoid(pred_clf_logit).numpy()
    pred_signal = np.where(pred_prob > 0.5, 1.0, -1.0)
    return idx_te, pred_signal


idx_te_num, signal_numeric = train_for_trading(NUMERIC_ONLY_FEATURES, last_fold, interp_seed)

# Reaproveita o modelo numérico+sentimento já treinado na Seção 10 (mesmo fold/seed)
with torch.no_grad():
    _, pred_clf_logit_sent = model_interp(torch.tensor(X_te_i))
pred_prob_sent = torch.sigmoid(pred_clf_logit_sent).numpy()
signal_sentiment = np.where(pred_prob_sent > 0.5, 1.0, -1.0)
idx_te_sent = idx_te_i

assert list(idx_te_num) == list(idx_te_sent), "Períodos de teste divergem entre os dois modelos — necessário para comparação de trading."


def simulate_strategy(signal: np.ndarray, log_returns: np.ndarray, cost_pct: float):
    signal = np.asarray(signal, dtype=float)
    position_prev = np.concatenate([[0.0], signal[:-1]])
    turnover = np.abs(signal - position_prev)
    strat_log_returns = signal * log_returns - turnover * cost_pct
    cum_curve = np.exp(np.cumsum(strat_log_returns))
    n_trades = int((turnover > 0).sum())
    total_cost = float((turnover * cost_pct).sum())
    return cum_curve, strat_log_returns, n_trades, total_cost


actual_log_returns = dataset.loc[idx_te_sent, "target_log_return"].values
cost_pct = CONFIG["transaction_cost_pct"]

cum_numeric, ret_numeric, n_trades_num, cost_num = simulate_strategy(signal_numeric, actual_log_returns, cost_pct)
cum_sentiment, ret_sentiment, n_trades_sent, cost_sent = simulate_strategy(signal_sentiment, actual_log_returns, cost_pct)
cum_buyhold = np.exp(np.cumsum(actual_log_returns))

trading_summary = pd.DataFrame([
    {"estratégia": "Numérico puro", "retorno_acumulado_%": (cum_numeric[-1] - 1) * 100,
     "n_trades": n_trades_num, "custo_total_%": cost_num * 100},
    {"estratégia": "Numérico + sentimento", "retorno_acumulado_%": (cum_sentiment[-1] - 1) * 100,
     "n_trades": n_trades_sent, "custo_total_%": cost_sent * 100},
    {"estratégia": "Buy-and-hold", "retorno_acumulado_%": (cum_buyhold[-1] - 1) * 100,
     "n_trades": 1, "custo_total_%": 0.0},
])
print(trading_summary.round(3).to_string(index=False))
trading_summary.to_csv(OUTPUTS_DIR / "trading_simulation_summary.csv", index=False)

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(idx_te_sent, cum_numeric, label="Numérico puro", color="#1f77b4")
ax.plot(idx_te_sent, cum_sentiment, label="Numérico + sentimento", color="#d62728")
ax.plot(idx_te_sent, cum_buyhold, label="Buy-and-hold", color="#7f7f7f", linestyle="--")
ax.set_title(f"Retorno acumulado — período de teste (fold {last_fold['fold']}), custo={cost_pct*100:.1f}%/operação")
ax.set_ylabel("Capital (base = 1.0)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
fig_path8 = FIGS_DIR / "08_trading_simulation.png"
plt.savefig(fig_path8, dpi=110)
plt.close(fig)
print("Figura salva em:", fig_path8)

# %% [markdown]
# ## 12. Discussão Final e Conclusões

# %%
print("### Resumo quantitativo automático (gerado a partir dos resultados desta execução) ###\n")
print(final_comparison.round(4).to_string(index=False))
print()
print(stat_test_df.round(4).to_string(index=False))
print()
n_sig = int(stat_test_df["significativo_5pct"].sum())
print(f"Métricas com diferença estatisticamente significativa (p<0.05) entre numérico puro "
      f"e numérico+sentimento: {n_sig} de {len(stat_test_df)}.")
best_trading = trading_summary.loc[trading_summary["retorno_acumulado_%"].idxmax(), "estratégia"]
print(f"Estratégia com maior retorno acumulado na simulação de trading: {best_trading}.")

# %% [markdown]
# ### 12.1 Interpretação à luz da EMH
#
# Sob a EMH semi-forte, notícias públicas já publicadas deveriam estar precificadas quase
# instantaneamente — logo, esperamos a priori pouco poder preditivo incremental do
# sentimento agregado sobre o fechamento seguinte.
#
# - **H0 não rejeitada:** consistente com a EMH semi-forte, e um resultado válido por si só —
#   contraria a narrativa (comercialmente promovida) de que "sentimento prevê o BTC de amanhã".
# - **Melhora significativa em alguma métrica:** não invalida a EMH sozinha — pode ser
#   ineficiência de curto prazo em cripto, informação além do fechamento UTC, ou overfitting
#   à amostra; por isso o teste pareado (múltiplas seeds/folds) e a simulação de trading com
#   custos são essenciais antes de qualquer conclusão prática.
# - Mesmo com ganho estatístico, sinais "picotados" (muitas trocas de posição) podem ser
#   destruídos pelo custo de transação — por isso a comparação sempre inclui buy-and-hold.
#
# ### 12.2 Limitações
# 1. Cobertura de notícias restrita a ~out/2021–dez/2023 (~33% do histórico de preço).
# 2. Fonte única de notícias — não cobre redes sociais (Twitter/X, Reddit, Telegram).
# 3. Granularidade diária; efeitos intradiários não capturados.
# 4. FinBERT não é específico para jargão cripto ("HODL", "rekt", "FUD"); CryptoBERT é
#    alternativa natural.
# 5. Amostra pequena (~800 dias) particionada em poucos folds — reduz poder estatístico.
# 6. Ativo único (BTC-USD); sentimento pode afetar altcoins de forma diferente.
#
# ### 12.3 Próximos passos
# - Horizontes adicionais (t+3, t+7) via `CONFIG["horizon_days"]`.
# - Sentimento social (Twitter/X, Reddit) com granularidade horária; CryptoBERT.
# - Ensemble de baselines e arquiteturas alternativas (Transformer leve, GRU).
# - Trading dimensionado pela confiança do modelo (Kelly fracionário), custos mais realistas.
# - Validar robustez em janelas de mercado distintas (bull/bear).

print("\nPipeline concluído. Ver outputs em:", OUTPUTS_DIR)
