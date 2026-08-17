"""Smoke-test da lógica de dev/pipeline.py com dados 100% sintéticos — sem downloads,
sem transformers/shap. Pega bugs de alinhamento/shapes/NaN rápido; não substitui a
execução real."""
import sys
import itertools
import numpy as np
import pandas as pd

np.random.seed(42)

print("=" * 70)
print("1) Indicadores técnicos")
print("=" * 70)

n_days = 400  # série de preço sintética
dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
price = 100 * np.exp(np.cumsum(np.random.normal(0, 0.02, n_days)))
volume = np.random.uniform(1e9, 5e9, n_days)
price_df = pd.DataFrame(
    {
        "Open": price * (1 + np.random.normal(0, 0.001, n_days)),
        "High": price * (1 + np.abs(np.random.normal(0, 0.005, n_days))),
        "Low": price * (1 - np.abs(np.random.normal(0, 0.005, n_days))),
        "Close": price,
        "Volume": volume,
    },
    index=dates,
)
price_df.index.name = "Date"
print("price_df shape:", price_df.shape)
assert price_df.isna().sum().sum() == 0


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

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    out["volume_change_1d"] = volume.pct_change(1)
    out["volume_ma_7"] = volume.rolling(7).mean()
    out["close"] = close
    return out


tech_features = compute_technical_features(price_df)
print("tech_features shape:", tech_features.shape)
print("NaN counts (esperado só no warm-up inicial):")
print(tech_features.isna().sum())
assert tech_features.iloc[35:].isna().sum().sum() == 0, "NaN inesperado além do warm-up!"
print("OK: indicadores técnicos sem NaN após warm-up (dia 35+).\n")

print("=" * 70)
print("2) Notícias sintéticas + alinhamento temporal + agregação")
print("=" * 70)

news_rows = []  # ~3 headlines/dia, cobrindo só metade do período (como no caso real)
news_start, news_end = dates[50], dates[300]
for d in pd.date_range(news_start, news_end, freq="D"):
    n_articles = np.random.poisson(3)
    for _ in range(n_articles):
        news_rows.append({
            "date": d + pd.Timedelta(hours=np.random.randint(0, 24)),
            "title": "headline sintética",
            "sent_score": np.clip(np.random.normal(0, 0.3), -1, 1),
            "sent_class": np.random.choice(["positive", "negative", "neutral"], p=[0.35, 0.25, 0.4]),
        })
news = pd.DataFrame(news_rows)
news["date"] = pd.to_datetime(news["date"], utc=True)
news["news_date"] = news["date"].dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None)
news["target_date"] = news["news_date"] + pd.Timedelta(days=1)
print("news shape:", news.shape, "| dias cobertos:", news["news_date"].nunique())


def aggregate_daily_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    g = news_df.groupby("news_date")
    agg = pd.DataFrame({
        "sent_mean": g["sent_score"].mean(),
        "sent_std": g["sent_score"].std(),
        "sent_median": g["sent_score"].median(),
        "news_count": g.size(),
    })
    class_dummies = pd.get_dummies(news_df["sent_class"])
    props = class_dummies.groupby(news_df["news_date"]).mean()
    props.columns = [f"sent_prop_{c}" for c in props.columns]
    agg = agg.join(props, how="left")
    return agg


daily_sent = aggregate_daily_sentiment(news)
daily_sent_full = daily_sent.reindex(price_df.index)
daily_sent_full["has_news"] = daily_sent_full["news_count"].notna().astype(int)
fill_zero_cols = [c for c in daily_sent_full.columns if c != "has_news"]
daily_sent_full[fill_zero_cols] = daily_sent_full[fill_zero_cols].fillna(0.0)
daily_sent_full["news_count"] = daily_sent_full["news_count"].astype(int)
daily_sent_full["sent_std"] = daily_sent_full["sent_std"].fillna(0.0)

assert daily_sent_full.isna().sum().sum() == 0, "Imputação neutra incompleta!"
print("daily_sent_full shape:", daily_sent_full.shape)
print("dias com has_news=1:", int(daily_sent_full["has_news"].sum()), "de", len(daily_sent_full))
# checagem: fora da janela de notícias, tudo deve ser neutro (0) com has_news=0
outside = daily_sent_full.loc[daily_sent_full.index < news_start]
assert (outside["has_news"] == 0).all() and (outside["sent_mean"] == 0).all()
print("OK: imputação neutra correta fora da janela de cobertura de notícias.\n")

print("=" * 70)
print("3) Merge, alvo (target) e checagem de look-ahead bias")
print("=" * 70)

feature_cols_tech = [c for c in tech_features.columns if c != "close"]
feature_cols_sent = list(daily_sent_full.columns)

features_full = tech_features[feature_cols_tech].join(daily_sent_full, how="left")
features_full["close"] = tech_features["close"]

H = 1
features_full["target_close"] = features_full["close"].shift(-H)
features_full["target_log_return"] = np.log(features_full["target_close"] / features_full["close"])
features_full["target_direction"] = (features_full["target_close"] > features_full["close"]).astype(int)

MODELING_START, MODELING_END = daily_sent.index.min(), daily_sent.index.max()
dataset = features_full.loc[MODELING_START:MODELING_END].dropna()
print("dataset shape:", dataset.shape)
assert dataset.isna().sum().sum() == 0

check = dataset.index + pd.Timedelta(days=H)
implied_target_close = price_df["Close"].reindex(check).values
assert np.allclose(dataset["target_close"].values, implied_target_close), "BUG de alinhamento!"
print(f"OK: target_close(t) == Close(t+{H}) para todas as {len(dataset)} linhas (sem look-ahead).\n")

FEATURE_COLUMNS = feature_cols_tech + feature_cols_sent

print("=" * 70)
print("4) Split walk-forward")
print("=" * 70)


def walk_forward_splits(index, n_folds, min_train_days, val_frac=0.15):
    n = len(index)
    assert n > min_train_days, "Dataset menor que o tamanho mínimo de treino."
    remaining = n - min_train_days
    fold_size = remaining // n_folds
    assert fold_size > 5, "Poucos dias por fold."
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


folds = walk_forward_splits(dataset.index, n_folds=3, min_train_days=100)  # dataset pequeno -> folds menores
for f in folds:
    print(f"  Fold {f['fold']}: train={len(f['train'])}d val={len(f['val'])}d test={len(f['test'])}d "
          f"| train_end={f['train'].max().date()} test_start={f['test'].min().date()}")
    assert f["train"].max() < f["val"].min() < f["test"].min()
    assert len(set(f["train"]) & set(f["val"]) & set(f["test"])) == 0
print("OK: folds estritamente cronológicos, sem overlap.\n")

print("=" * 70)
print("5) Baselines: persistência + métricas")
print("=" * 70)

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,
    accuracy_score, f1_score, confusion_matrix, roc_auc_score,
)


def regression_metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def classification_metrics(y_true, y_pred, y_score=None):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    if y_score is not None and len(np.unique(y_true)) > 1:
        out["auc"] = float(roc_auc_score(y_true, y_score))
    else:
        out["auc"] = float("nan")
    return out


test_idx = folds[-1]["test"]
close_t = dataset.loc[test_idx, "close"]
target_close = dataset.loc[test_idx, "target_close"]
pred_close = close_t.copy()
prev_direction = (dataset.loc[test_idx, "log_return_1d"] > 0).astype(int)
true_direction = dataset.loc[test_idx, "target_direction"]

reg_m = regression_metrics(target_close, pred_close)
clf_m = classification_metrics(true_direction, prev_direction, y_score=prev_direction.astype(float))
print("Persistência ingênua:", reg_m, "|", {k: v for k, v in clf_m.items() if k != "confusion_matrix"})
assert reg_m["RMSE"] > 0 and 0 <= clf_m["accuracy"] <= 1
print("OK: baseline de persistência roda e produz métricas plausíveis.\n")

print("=" * 70)
print("6) ARIMA (statsmodels) — fit rápido em série sintética pequena")
print("=" * 70)
from statsmodels.tsa.arima.model import ARIMA

train_log_close = np.log(dataset.loc[folds[-1]["train"], "close"])
fit = ARIMA(train_log_close, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit()
fc = fit.forecast(steps=3)
print("ARIMA ajustou e gerou forecast de 3 passos:", fc.values)
assert len(fc) == 3 and not np.isnan(fc).any()
print("OK: statsmodels ARIMA funciona como esperado (API idêntica à usada no pipeline real).\n")

print("=" * 70)
print("7) Sequências para LSTM + smoke-test de treino (torch)")
print("=" * 70)
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


def fit_scaler_and_transform(fold, feature_cols, dataset_):
    scaler = StandardScaler().fit(dataset_.loc[fold["train"], feature_cols])
    scaled = pd.DataFrame(scaler.transform(dataset_[feature_cols]), index=dataset_.index, columns=feature_cols)
    return scaler, scaled


def build_sequences(scaled_features, target_idx, seq_len, dataset_):
    full_idx = scaled_features.index
    pos_map = {d: i for i, d in enumerate(full_idx)}
    X_list, y_reg_list, y_clf_list, kept_idx = [], [], [], []
    for d in target_idx:
        pos = pos_map[d]
        if pos - seq_len + 1 < 0:
            continue
        window = scaled_features.iloc[pos - seq_len + 1: pos + 1].values
        X_list.append(window)
        y_reg_list.append(dataset_.loc[d, "target_log_return"])
        y_clf_list.append(dataset_.loc[d, "target_direction"])
        kept_idx.append(d)
    X = np.stack(X_list).astype(np.float32)
    y_reg = np.array(y_reg_list, dtype=np.float32)
    y_clf = np.array(y_clf_list, dtype=np.float32)
    return X, y_reg, y_clf, pd.DatetimeIndex(kept_idx)


class LSTMForecaster(nn.Module):
    def __init__(self, n_features, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=num_layers, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.reg_head = nn.Linear(hidden_size, 1)
        self.clf_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.reg_head(last).squeeze(-1), self.clf_head(last).squeeze(-1)


seq_len = 10
fold0 = folds[0]
scaler, scaled = fit_scaler_and_transform(fold0, FEATURE_COLUMNS, dataset)
X_tr, yreg_tr, yclf_tr, idx_tr = build_sequences(scaled, fold0["train"], seq_len, dataset)
X_val, yreg_val, yclf_val, idx_val = build_sequences(scaled, fold0["val"], seq_len, dataset)
X_te, yreg_te, yclf_te, idx_te = build_sequences(scaled, fold0["test"], seq_len, dataset)
print(f"X_tr={X_tr.shape} X_val={X_val.shape} X_te={X_te.shape} (n_features={len(FEATURE_COLUMNS)})")
assert X_tr.shape[1] == seq_len and X_tr.shape[2] == len(FEATURE_COLUMNS)
assert not np.isnan(X_tr).any() and not np.isnan(yreg_tr).any()

torch.manual_seed(0)
model = LSTMForecaster(len(FEATURE_COLUMNS), hidden_size=8, num_layers=1)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()

X_tr_t = torch.tensor(X_tr)
yreg_tr_t = torch.tensor(yreg_tr)
yclf_tr_t = torch.tensor(yclf_tr)

losses = []
for epoch in range(3):  # smoke-test: só 3 épocas, dataset minúsculo
    model.train()
    opt.zero_grad()
    pred_reg, pred_clf = model(X_tr_t)
    loss = mse(pred_reg, yreg_tr_t) + bce(pred_clf, yclf_tr_t)
    loss.backward()
    opt.step()
    losses.append(loss.item())
print("losses por época (deve ser finito e idealmente decrescente):", losses)
assert all(np.isfinite(losses)), "Loss não-finita — bug na arquitetura/perda!"

model.eval()
with torch.no_grad():
    pred_reg_te, pred_clf_te = model(torch.tensor(X_te))
pred_clf_prob = torch.sigmoid(pred_clf_te).numpy()
assert pred_reg_te.shape[0] == X_te.shape[0]
assert ((pred_clf_prob >= 0) & (pred_clf_prob <= 1)).all()
print("OK: forward/backward do LSTM com 2 cabeças roda sem erro, shapes corretos.\n")

print("=" * 70)
print("8) Permutation importance — smoke-test")
print("=" * 70)


def evaluate_lstm(model, X_test, y_reg_test, y_clf_test, close_t_test, target_close_true):
    with torch.no_grad():
        pred_reg, pred_clf_logit = model(torch.tensor(X_test))
    pred_reg = pred_reg.numpy()
    pred_clf_prob = torch.sigmoid(pred_clf_logit).numpy()
    pred_clf_label = (pred_clf_prob > 0.5).astype(int)
    pred_close = close_t_test * np.exp(pred_reg)
    reg_m = regression_metrics(target_close_true, pred_close)
    clf_m = classification_metrics(y_clf_test.astype(int), pred_clf_label, y_score=pred_clf_prob)
    return reg_m, clf_m, pred_reg, pred_clf_prob


close_t_te = dataset.loc[idx_te, "close"].values
target_close_te = dataset.loc[idx_te, "target_close"].values


def permutation_importance_lstm(model, X_test, y_reg_test, y_clf_test, close_t_test,
                                 target_close_test, feature_names, seed=0, n_repeats=2):
    rng = np.random.default_rng(seed)
    base_reg, base_clf, _, _ = evaluate_lstm(model, X_test, y_reg_test, y_clf_test, close_t_test, target_close_test)
    rows = []
    n_samples = X_test.shape[0]
    for j, fname in enumerate(feature_names):
        rmse_deltas = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            perm_idx = rng.permutation(n_samples)
            X_perm[:, :, j] = X_perm[perm_idx, :, j]
            reg_m, _, _, _ = evaluate_lstm(model, X_perm, y_reg_test, y_clf_test, close_t_test, target_close_test)
            rmse_deltas.append(reg_m["RMSE"] - base_reg["RMSE"])
        rows.append({"feature": fname, "delta_RMSE": float(np.mean(rmse_deltas))})
    return pd.DataFrame(rows).sort_values("delta_RMSE", ascending=False)


perm_df = permutation_importance_lstm(model, X_te, yreg_te, yclf_te, close_t_te, target_close_te,
                                       FEATURE_COLUMNS, n_repeats=2)
print(perm_df.head(5).to_string(index=False))
assert perm_df.shape[0] == len(FEATURE_COLUMNS)
print("OK: permutation importance roda sem erro para todas as features.\n")

print("=" * 70)
print("9) Simulação de trading")
print("=" * 70)


def simulate_strategy(signal, log_returns, cost_pct):
    signal = np.asarray(signal, dtype=float)
    position_prev = np.concatenate([[0.0], signal[:-1]])
    turnover = np.abs(signal - position_prev)
    strat_log_returns = signal * log_returns - turnover * cost_pct
    cum_curve = np.exp(np.cumsum(strat_log_returns))
    return cum_curve, strat_log_returns, int((turnover > 0).sum()), float((turnover * cost_pct).sum())


signal = np.where(pred_clf_prob > 0.5, 1.0, -1.0)
actual_returns = dataset.loc[idx_te, "target_log_return"].values
cum, ret, n_trades, cost = simulate_strategy(signal, actual_returns, 0.001)
print(f"retorno acumulado final: {(cum[-1]-1)*100:.2f}% | n_trades={n_trades} | custo_total={cost*100:.3f}%")
assert np.isfinite(cum).all()
print("OK: simulação de trading roda sem erro.\n")

print("=" * 70)
print("10) Teste estatístico pareado (scipy)")
print("=" * 70)
from scipy import stats

a = np.random.normal(0.05, 0.01, 15)  # simula 3 folds x 5 seeds
b = a + np.random.normal(-0.002, 0.005, 15)
t_stat, t_p = stats.ttest_rel(b, a)
w_stat, w_p = stats.wilcoxon(b, a)
print(f"t-test: stat={t_stat:.3f} p={t_p:.4f} | wilcoxon: stat={w_stat:.3f} p={w_p:.4f}")
assert 0 <= t_p <= 1 and 0 <= w_p <= 1
print("OK: testes estatísticos pareados rodam corretamente.\n")

print("=" * 70)
print("11) PCA exploratório")
print("=" * 70)
from sklearn.decomposition import PCA

scaler_pca = StandardScaler().fit(dataset[FEATURE_COLUMNS])
X_pca_scaled = scaler_pca.transform(dataset[FEATURE_COLUMNS])
pca = PCA(n_components=min(10, X_pca_scaled.shape[1])).fit(X_pca_scaled)
print("Variância explicada (top 5):", np.round(pca.explained_variance_ratio_[:5], 3))
assert np.isclose(pca.explained_variance_ratio_.sum(), 1.0, atol=0.3) or pca.explained_variance_ratio_.sum() <= 1.0
print("OK: PCA roda sem erro.\n")

print("=" * 70)
print("TODOS OS TESTES DE LÓGICA PASSARAM (dados sintéticos, sem downloads/instalações pesadas).")
print("=" * 70)
