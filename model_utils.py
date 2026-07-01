import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Simple MLP Regression Model
# ============================================================
class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Local Training (batched)
# ============================================================
def train_local_batched(model, X, y, epochs=5, lr=0.001, batch_size=32):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    return model


def train_local_fedprox(model, X, y, global_state, mu, epochs=5, lr=0.001, batch_size=32):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

    # Save global weights for proximal term
    global_params = {k: v.clone().detach() for k, v in global_state.items()}

    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()

            preds = model(xb)
            loss = criterion(preds, yb)

            # === FedProx Proximal Term ===
            prox = 0.0
            for (name, param) in model.named_parameters():
                prox += ((param - global_params[name]) ** 2).sum()

            loss += (mu / 2.0) * prox

            loss.backward()
            optimizer.step()

    return model


# ============================================================
# Evaluation
# ============================================================
def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        preds = model(X).numpy()
        y_np = y.numpy()

    mse = mean_squared_error(y_np, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_np, preds)
    r2 = r2_score(y_np, preds)

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


# ============================================================
# FEDAVG — Uniform equal-weight aggregation
# ============================================================
def fedavg(models):
    """
    Standard FedAvg: simple uniform average of all model parameters.

    Args:
        models : list of nn.Module

    Returns:
        state_dict of uniformly averaged model
    """
    new_state = {}
    keys = models[0].state_dict().keys()

    for key in keys:
        tensors = [m.state_dict()[key].float() for m in models]
        new_state[key] = sum(tensors) / len(tensors)

    return new_state


# ============================================================
# GADFL — Geography-Aware Weighted Aggregation
# ============================================================
def weighted_fedavg(models, weights):
    """
    Geography-aware weighted average of client models.

    Used by GADFL instead of uniform FedAvg. Agents with similar
    geographic environments (low JSD) receive higher weights.

    Args:
        models  : list of nn.Module in the same order as weights
        weights : list of float weights (same order as models)
                  does NOT need to sum to 1 — normalized internally

    Returns:
        state_dict of weighted average model

    Example:
        # Agent 1 (urban) trusts Agent 2 (urban) more than Agent 3 (highway)
        models  = [model_1, model_2, model_3]
        weights = [0.31,    0.24,    0.13   ]   # from JSD computation
        result  = weighted_fedavg(models, weights)
        # = 0.31*model_1 + 0.24*model_2 + 0.13*model_3  (after renorm to 1.0)
    """
    if len(models) != len(weights):
        raise ValueError(
            f"weighted_fedavg: {len(models)} models but {len(weights)} weights. "
            f"They must have the same length."
        )

    if len(models) == 0:
        raise ValueError("weighted_fedavg: empty models list")

    # Convert to numpy and normalize to sum=1
    w = np.array(weights, dtype=np.float64)
    total = w.sum()

    if total < 1e-10:
        # Fallback to uniform if all weights are zero
        w = np.ones(len(models), dtype=np.float64) / len(models)
    else:
        w = w / total

    new_state = {}
    keys = models[0].state_dict().keys()

    for key in keys:
        tensors = [m.state_dict()[key].float() for m in models]
        # Weighted sum: Σ w_i * θ_i
        weighted_sum = sum(float(wi) * ti for wi, ti in zip(w, tensors))
        new_state[key] = weighted_sum

    return new_state


# ============================================================
# INCREMENTAL CLIENT SCALING — INITIAL WEIGHT MANAGEMENT
# ============================================================

def save_initial_weights(client_id, state_dict, init_dir="initial_weights"):
    """
    Persist the initial model weights for a client to the shared init directory.

    This is called ONCE per client — the very first time that client ID appears
    in any experiment. On all future runs, that client will call
    load_initial_weights() instead of regenerating weights.

    The init_dir is intentionally kept OUTSIDE any algorithm subfolder
    so the weights are shared across all algorithm variants and scaling steps.
    """
    os.makedirs(init_dir, exist_ok=True)
    path = os.path.join(init_dir, f"client{client_id}_init.pt")
    torch.save(state_dict, path)
    print(
        f"[ScalingInit] Client {client_id}: "
        f"NEW initial weights saved → {path}"
    )


def load_initial_weights(client_id, init_dir="initial_weights"):
    """
    Load previously saved initial weights for a client.

    Returns the state_dict if the file exists, or None if this is a new client.
    """
    path = os.path.join(init_dir, f"client{client_id}_init.pt")
    if not os.path.exists(path):
        return None
    state_dict = torch.load(path, map_location="cpu")
    print(
        f"[ScalingInit] Client {client_id}: "
        f"Loaded EXISTING initial weights from {path}"
    )
    return state_dict


def load_or_init_weights(client_id, input_dim, init_dir="initial_weights", device=None):
    """
    Core function for incremental client scaling.

    Decision logic:
      ┌─────────────────────────────────────────────────────┐
      │  Does  init_dir/client{id}_init.pt  exist?          │
      │                                                      │
      │  YES  →  Load weights (existing client).            │
      │          Do NOT touch PyTorch random state.         │
      │                                                      │
      │  NO   →  Initialise weights with current RNG state  │
      │          (new client, seed already set by caller).  │
      │          Save the result so future runs reuse it.   │
      └─────────────────────────────────────────────────────┘
    """
    if device is None:
        device = torch.device("cpu")

    existing_state = load_initial_weights(client_id, init_dir)

    if existing_state is not None:
        model = RegressionModel(input_dim).to(device)
        model.load_state_dict(existing_state)
        print(
            f"[ScalingInit] Client {client_id}: "
            f"Re-using saved initial weights (existing client)."
        )
    else:
        model = RegressionModel(input_dim).to(device)
        save_initial_weights(client_id, model.state_dict(), init_dir)
        print(
            f"[ScalingInit] Client {client_id}: "
            f"First-time initialisation — weights saved for future reuse."
        )

    return model
