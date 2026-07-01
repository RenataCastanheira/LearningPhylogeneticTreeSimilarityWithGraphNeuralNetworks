import numpy as np
import torch
from sklearn.metrics import mean_absolute_percentage_error, r2_score

# --------------------------------------------------------------------------- #
# Evaluation (always in the original SPR scale)
# --------------------------------------------------------------------------- #
def evaluate(model, loader, device, log_target: bool):
    """
    Run the model on `loader`, return (true_y, predicted_y) in raw scale.
    """
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for batch_a, batch_b, y in loader:
            batch_a = batch_a.to(device)
            batch_b = batch_b.to(device)
            yhat = model(batch_a, batch_b)
            yhat = torch.expm1(yhat) if log_target else yhat
            yhat = yhat.clamp(min=0.0)         # SPR distance is non-negative
            preds.extend(yhat.cpu().tolist())
            tgts.extend(y.cpu().tolist())     # `y` already in raw scale here
    return np.asarray(tgts, dtype=float), np.asarray(preds, dtype=float)


def metrics(t: np.ndarray, p: np.ndarray):
    mae = float(np.mean(np.abs(t - p)))
    rmse = float(np.sqrt(np.mean((t - p) ** 2)))
    r2 = float(r2_score(t, p)) if len(t) > 1 else float("nan")
    nz = t > 0
    mape = (
        float(mean_absolute_percentage_error(t[nz], p[nz])) if nz.any() else float("nan")
    )
    return mae, rmse, r2, mape

