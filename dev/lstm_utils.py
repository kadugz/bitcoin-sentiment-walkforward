"""Modelo LSTM, splits e treino — reaproveitado de dev/pipeline.py, usado por
dev/make_extra_figures.py e dev/regenerate_figures.py."""
import os
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGS_DIR = PROJECT_ROOT / "figs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

CONFIG = {
    "sequence_length": 30,
    "lstm_hidden_size": 32,
    "lstm_num_layers": 1,
    "lstm_dropout": 0.0,
    "lstm_lr": 1e-3,
    "lstm_batch_size": 32,
    "lstm_max_epochs": 100,
    "lstm_patience": 10,
    "seed": 0,
    "n_walk_forward_folds": 5,
    "min_train_days": 365,
    "transaction_cost_pct": 0.001,
}

FEATURE_COLS_TECH = [
    "log_return_1d", "log_return_3d", "log_return_7d", "ma_7", "price_to_ma_7",
    "ma_14", "price_to_ma_14", "ma_30", "price_to_ma_30", "volatility_7",
    "volatility_14", "rsi_14", "volume_change_1d", "volume_ma_7",
]
FEATURE_COLS_SENT = [
    "sent_mean", "sent_std", "sent_median", "news_count", "sent_prop_negative",
    "sent_prop_neutral", "sent_prop_positive", "has_news",
]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def walk_forward_splits(index: pd.DatetimeIndex, n_folds: int, min_train_days: int, val_frac: float = 0.15):
    n = len(index)
    remaining = n - min_train_days
    fold_size = remaining // n_folds
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


def fit_scaler_and_transform(train_idx, full_df, feature_cols):
    scaler = StandardScaler().fit(full_df.loc[train_idx, feature_cols])
    scaled = pd.DataFrame(scaler.transform(full_df[feature_cols]), index=full_df.index, columns=feature_cols)
    return scaler, scaled


def build_sequences(scaled_features, target_idx, seq_len, full_df,
                     target_cols=("target_log_return", "target_direction")):
    full_idx = scaled_features.index
    pos_map = {d: i for i, d in enumerate(full_idx)}
    X_list, y_reg_list, y_clf_list, kept_idx = [], [], [], []
    for d in target_idx:
        pos = pos_map[d]
        if pos - seq_len + 1 < 0:
            continue
        window = scaled_features.iloc[pos - seq_len + 1: pos + 1].values
        X_list.append(window)
        y_reg_list.append(full_df.loc[d, target_cols[0]])
        y_clf_list.append(full_df.loc[d, target_cols[1]])
        kept_idx.append(d)
    X = np.stack(X_list).astype(np.float32)
    y_reg = np.array(y_reg_list, dtype=np.float32)
    y_clf = np.array(y_clf_list, dtype=np.float32)
    return X, y_reg, y_clf, pd.DatetimeIndex(kept_idx)


def train_lstm(X_train, y_reg_train, y_clf_train, X_val, y_reg_val, y_clf_val, n_features, seed):
    set_global_seed(seed)
    model = LSTMForecaster(n_features, CONFIG["lstm_hidden_size"], CONFIG["lstm_num_layers"], CONFIG["lstm_dropout"])
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


def evaluate_close_predictions(model, X_test, close_t):
    with torch.no_grad():
        pred_reg, pred_clf_logit = model(torch.tensor(X_test))
    pred_reg = pred_reg.numpy()
    pred_prob = torch.sigmoid(pred_clf_logit).numpy()
    pred_close = close_t * np.exp(pred_reg)
    return pred_close, pred_reg, pred_prob
