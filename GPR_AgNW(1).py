"""
GPR-based multi-objective Bayesian optimization for AgNW spray-coating.

SPDX-License-Identifier: CC-BY-4.0
This work is licensed under the Creative Commons Attribution 4.0 International License.
To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/

This script implements Gaussian process regression models to predict optical
transmittance and sheet resistance, and performs Pareto front analysis to
identify optimal processing parameters.

Author: Polina S. Mikhailova et al.
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

# SETTINGS

PARAM_BOUNDS = {
    'power': (0.05, 1.0),
    'mass_flow': (100, 300),
    'PVP_conc': (0.0, 0.5)
}
TARGET_REGION = {'transparency_min': 75.0, 'resistance_max': 10.0}

N_CANDIDATES = 20000
N_SELECT_ON_FRONT = 3
N_LOCAL = 2
LOCAL_PCT = 0.05

# HELPER FUNCTIONS

def sample_candidates(bounds, n):
    """Generate random candidate points within parameter bounds."""
    keys = list(bounds.keys())
    arr = np.zeros((n, len(keys)))
    for i, k in enumerate(keys):
        lo, hi = bounds[k]
        arr[:, i] = np.random.uniform(lo, hi, size=n)
    return arr, keys


def is_pareto_efficient(costs):
    """Identify Pareto-efficient points."""
    N = costs.shape[0]
    is_pareto = np.ones(N, dtype=bool)
    for i in range(N):
        if not is_pareto[i]:
            continue
        for j in range(N):
            if i == j:
                continue
            if (costs[j, 0] >= costs[i, 0] and costs[j, 1] <= costs[i, 1]) and \
               ((costs[j, 0] > costs[i, 0]) or (costs[j, 1] < costs[i, 1])):
                is_pareto[i] = False
                break
    return is_pareto


def select_on_front(pareto_X, pareto_Y, keys, target=None, n_select=5):
    """Select points closest to target region from Pareto front."""
    if len(pareto_X) == 0:
        return np.empty((0, pareto_X.shape[1]))
    if target:
        tgt = np.array([target['transparency_min'], target['resistance_max']])
        scale = np.array([10.0, 5.0])
        dists = np.linalg.norm((pareto_Y - tgt.reshape(1, 2)) / scale, axis=1)
        idx = np.argsort(dists)[:n_select]
        return pareto_X[idx]
    else:
        order = np.argsort(pareto_Y[:, 0])
        ordered_X = pareto_X[order]
        if n_select >= len(ordered_X):
            return ordered_X
        idxs = np.linspace(0, len(ordered_X) - 1, n_select).astype(int)
        return ordered_X[idxs]


def make_local_variations(x, bounds, n_local=4, pct=0.05):
    """Generate local variations around a point."""
    lo = np.array([bounds[k][0] for k in bounds])
    hi = np.array([bounds[k][1] for k in bounds])
    span = hi - lo
    res = []
    for _ in range(n_local):
        perturb = (np.random.uniform(-1, 1, size=x.shape) * pct) * span
        new = np.minimum(np.maximum(x + perturb, lo), hi)
        res.append(new)
    return np.vstack(res)


def run_optimization_iteration(X_train, y_trans, y_res, bounds, keys,
                               n_candidates=N_CANDIDATES,
                               n_select=N_SELECT_ON_FRONT,
                               n_local=N_LOCAL, local_pct=LOCAL_PCT):
    """
    Perform one iteration of Bayesian optimization.

    Returns:
        new_points: Suggested experimental points
        cand_df: DataFrame with predictions for all candidates
        pareto_X: Parameters of Pareto-optimal points
        mean_sigma: Mean predictive uncertainty
    """
    # Normalization
    xscaler = StandardScaler().fit(X_train)
    Xs = xscaler.transform(X_train)
    yscaler_t = StandardScaler().fit(y_trans.reshape(-1, 1))
    yscaler_r = StandardScaler().fit(y_res.reshape(-1, 1))
    ys_t_s = yscaler_t.transform(y_trans.reshape(-1, 1)).ravel()
    ys_r_s = yscaler_r.transform(y_res.reshape(-1, 1)).ravel()

    # GPR models with Rational Quadratic kernel
    kernel = C(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=1.0, alpha=1.0)
    gp_t = GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                    n_restarts_optimizer=5)
    gp_r = GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                    n_restarts_optimizer=5)
    gp_t.fit(Xs, ys_t_s)
    gp_r.fit(Xs, ys_r_s)

    # Generate candidate points
    candidates, _ = sample_candidates(bounds, n_candidates)
    cand_s = xscaler.transform(candidates)
    mu_t_s, sigma_t_s = gp_t.predict(cand_s, return_std=True)
    mu_r_s, sigma_r_s = gp_r.predict(cand_s, return_std=True)
    mu_t = yscaler_t.inverse_transform(mu_t_s.reshape(-1, 1)).ravel()
    mu_r = yscaler_r.inverse_transform(mu_r_s.reshape(-1, 1)).ravel()
    sigma_t = sigma_t_s * yscaler_t.scale_[0]
    sigma_r = sigma_r_s * yscaler_r.scale_[0]

    # Conservative (LCB) Pareto front
    kappa = 1.0
    eff_trans = mu_t - kappa * sigma_t
    eff_res = mu_r + kappa * sigma_r
    preds = np.vstack([eff_trans, eff_res]).T
    pareto_mask = is_pareto_efficient(preds)
    pareto_X = candidates[pareto_mask]
    pareto_Y = preds[pareto_mask]

    # Select points for next experiment
    selected_on_front = select_on_front(pareto_X, pareto_Y, keys,
                                        target=TARGET_REGION,
                                        n_select=n_select)

    # Local variations
    new_points = []
    for x in selected_on_front:
        new_points.append(x)
        local = make_local_variations(x, bounds, n_local=n_local, pct=local_pct)
        for p in local:
            new_points.append(p)
    new_points = np.vstack(new_points) if len(new_points) > 0 else np.empty((0, len(keys)))

    # Exploration points
    explore_idx = np.argsort(-(sigma_t + sigma_r))[:2]
    new_points = np.vstack([new_points, candidates[explore_idx]])
    new_points = np.unique(new_points, axis=0)
    new_points = np.array([x for x in new_points if
                           np.min(np.linalg.norm(X_train - x, axis=1)) > 1e-6])

    # Save candidate predictions
    cand_df = pd.DataFrame(candidates, columns=keys)
    cand_df['mu_trans'] = mu_t
    cand_df['mu_res'] = mu_r
    cand_df['sigma_trans'] = sigma_t
    cand_df['sigma_res'] = sigma_r
    cand_df['pareto'] = pareto_mask.astype(int)

    mean_sigma = np.mean(sigma_t + sigma_r)

    return new_points, cand_df, pareto_X, mean_sigma

# EXAMPLE USAGE
# Note: This runs one iteration of the optimization.
# For full iterative optimization, call this function multiple times with updated training data after each experiment.

if __name__ == "__main__":
    # Load experimental data
    df = pd.read_csv("my_data.csv")
    X_train = df[['power', 'mass_flow', 'PVP_conc']].values
    y_trans = df['transparency'].values
    y_res = df['resistance'].values

    keys = list(PARAM_BOUNDS.keys())

    # Run one optimization iteration
    new_points, cand_df, pareto_X, mean_sigma = run_optimization_iteration(
        X_train, y_trans, y_res, PARAM_BOUNDS, keys
    )

    # Save results
    pd.DataFrame(new_points, columns=keys).to_csv("proposed_points.csv", index=False)
    cand_df.to_csv("candidate_predictions.csv", index=False)