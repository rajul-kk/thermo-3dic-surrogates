"""Per-sample S5 sufficient statistics for the power-scaling rule (T_amb + mean S4 rise x P/mean P, section 9.29).
Writes results/icb_ci/persample_s5_powerscale.npz (sse, sy, syy per S5 sample, manifest case ids)."""
from pathlib import Path
import json
import numpy as np
from scripts.ic_thermbench_data import SCOPES, load_scope

root = Path('data/ic-thermbench/datasets')
s4 = load_scope(root, 'level4', fix_orientation=True)
ch = SCOPES['level4']; ip, ia = ch.index('chiplet_power'), ch.index('ambient_K')
def stats(x, y):
    P = x[..., ip].reshape(len(x), -1).sum(1).astype(np.float64)
    A = x[..., ia].reshape(len(x), -1).mean(1).astype(np.float64)
    return P, A
P4, A4 = stats(s4.x_train, s4.y_train)
rise = (s4.y_train.astype(np.float64) - A4[:, None, None, None]).mean(0)
Pm = P4.mean()
s5 = load_scope(root, 'level5', split_data=False, fix_orientation=True)
P5, A5 = stats(s5.x_test, s5.y_test)
y = s5.y_test.astype(np.float64).reshape(len(P5), -1)
pred = (A5[:, None, None, None] + rise[None] * (P5 / Pm)[:, None, None, None]).reshape(len(P5), -1)
sse = ((pred - y) ** 2).sum(1)
print('n', len(y), 'RMSE', np.sqrt(sse.sum() / y.size), 'R2', 1 - sse.sum() / ((y - y.mean()) ** 2).sum())
np.savez('results/icb_ci/persample_s5_powerscale.npz', sse=sse, sy=y.sum(1), syy=(y ** 2).sum(1), n_cells=y.shape[1])
