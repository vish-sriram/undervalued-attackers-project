"""Fit and audit the fixed four-predictor transfer fee model."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

root = Path(__file__).resolve().parent
source = root / 'outputs/attacker_history/model_ready_features.csv'
out = root / 'outputs/fee_model'
out.mkdir(parents=True, exist_ok=True)
data = pd.read_csv(source)
columns = ['age_at_transfer', 'years_until_selling_contract_expiry', 'total_minutes', 'xg_per90']
x = data[columns].to_numpy(dtype=float)
assert (x[:, 2] > 0).all()
x[:, 2] = np.log(x[:, 2])
terms = ['intercept', 'age_at_transfer', 'years_until_selling_contract_expiry', 'ln_prior_season_minutes', 'xg_per90']
fee = data.transfer_fee_eur.to_numpy(dtype=float) / 1e6
train = data.window.eq('summer_2025').to_numpy()
test = data.window.eq('summer_2026').to_numpy()
assert (train.sum(), test.sum(), len(data)) == (21, 16, 37)
assert data.player_id.is_unique and (fee > 0).all() and np.isfinite(x).all()

def model(alpha=1):
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), Ridge(alpha=alpha)),
        func=np.log, inverse_func=np.exp,
    )

def metrics(actual, predicted):
    log_actual, log_predicted = np.log(actual), np.log(predicted)
    return {
        'n': len(actual),
        'mae_eur_m': mean_absolute_error(actual, predicted),
        'median_absolute_error_eur_m': median_absolute_error(actual, predicted),
        'rmse_eur_m': np.sqrt(np.mean((actual - predicted)**2)),
        'bias_pred_minus_actual_eur_m': np.mean(predicted - actual),
        'pearson_r_euro': np.corrcoef(actual, predicted)[0, 1] if np.ptp(predicted) else None,
        'r_squared_euro': r2_score(actual, predicted),
        'pearson_r_log': np.corrcoef(log_actual, log_predicted)[0, 1] if np.ptp(log_predicted) else None,
        'r_squared_log': r2_score(log_actual, log_predicted),
    }

grid = [0.01, 0.1, 1, 10, 100]
search = GridSearchCV(model(), {'regressor__ridge__alpha': grid}, cv=LeaveOneOut(),
                      scoring='neg_mean_absolute_error').fit(x[train], fee[train])
alpha = search.best_params_['regressor__ridge__alpha']
pred = search.predict(x[test])
final = model(alpha).fit(x, fee)
pipe = final.regressor_
scaler, ridge = pipe.named_steps['standardscaler'], pipe.named_steps['ridge']
raw = ridge.coef_ / scaler.scale_
intercept = ridge.intercept_ - scaler.mean_ @ raw
final_pred = final.predict(x)
np.testing.assert_allclose(np.exp(intercept + x @ raw), final_pred, rtol=1e-12)
z = scaler.transform(x)
np.testing.assert_allclose(
    np.linalg.solve(z.T @ z + alpha * np.eye(x.shape[1]), z.T @ (np.log(fee) - np.log(fee).mean())),
    ridge.coef_, rtol=1e-10, atol=1e-12)

# Auxiliary, unpenalized OLS inference using HC3 and a t reference.
design = np.column_stack([np.ones(len(x)), x])
log_fee = np.log(fee)
ols = np.linalg.lstsq(design, log_fee, rcond=None)[0]
bread = np.linalg.inv(design.T @ design)
residuals = log_fee - design @ ols
leverage = np.einsum('ij,jk,ik->i', design, bread, design)
adjusted = design * (residuals / (1 - leverage))[:, None]
cov = bread @ (adjusted.T @ adjusted) @ bread
se = np.sqrt(cov.diagonal())
df = len(x) - design.shape[1]
t = ols / se
p = 2 * stats.t.sf(abs(t), df)
q = stats.t.ppf(0.975, df)
f_stat = ols[1:] @ np.linalg.solve(cov[1:, 1:], ols[1:]) / len(raw)

# Resample each window separately. These intervals describe fixed-alpha stability.
rng = np.random.default_rng(2026)
indices = [np.flatnonzero(mask) for mask in (train, test)]
bootstrap = []
for _ in range(5000):
    i = np.concatenate([rng.choice(group, len(group), replace=True) for group in indices])
    mean, scale = x[i].mean(axis=0), x[i].std(axis=0)
    scale[scale == 0] = 1
    zz = (x[i] - mean) / scale
    yy = log_fee[i]
    slopes = np.linalg.solve(zz.T @ zz + alpha * np.eye(len(raw)), zz.T @ (yy - yy.mean())) / scale
    bootstrap.append(np.r_[yy.mean() - mean @ slopes, slopes])
lo, hi = np.quantile(bootstrap, [0.025, 0.975], axis=0)
coefficients = pd.DataFrame({
    'term': terms, 'ridge_coefficient': np.r_[intercept, raw],
    'ridge_standardized_coefficient': np.r_[ridge.intercept_, ridge.coef_],
    'ridge_bootstrap_95_low': lo, 'ridge_bootstrap_95_high': hi,
    'ols_coefficient': ols, 'ols_hc3_se': se, 'ols_hc3_t': t, 'ols_hc3_p': p,
    'ols_hc3_95_low': ols - q*se, 'ols_hc3_95_high': ols + q*se,
})
coefficients.to_csv(out / 'coefficients.csv', index=False)
predictions = data.loc[test, ['player_id','name','window','transfer_fee_eur']].copy()
predictions['predicted_fee_eur'] = pred * 1e6
predictions['error_pred_minus_actual_eur'] = predictions.predicted_fee_eur - predictions.transfer_fee_eur
predictions.to_csv(out / 'holdout_predictions.csv', index=False)
summary = {
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'target': 'natural log of transfer fee in millions of euros',
    'alpha': alpha, 'training_window': 'summer_2025', 'holdout_window': 'summer_2026',
    'training_loo_mae_eur_m_by_alpha': dict(zip(map(str, grid), -search.cv_results_['mean_test_score'])),
    'holdout': metrics(fee[test], pred),
    'holdout_2025_median_baseline': metrics(fee[test], np.full(test.sum(), np.median(fee[train]))),
    'final_refit_in_sample': metrics(fee, final_pred),
    'final_model': {'terms': terms, 'coefficients': [intercept, *raw],
                    'feature_means': scaler.mean_.tolist(), 'feature_scales': scaler.scale_.tolist()},
    'auxiliary_ols': {'log_r_squared': r2_score(log_fee, design @ ols),
                      'log_adjusted_r_squared': 1-(1-r2_score(log_fee, design @ ols))*(len(x)-1)/df,
                      'hc3_joint_f': f_stat, 'f_df': [len(raw), df],
                      'hc3_joint_p': stats.f.sf(f_stat, len(raw), df)},
}
(out / 'model_and_metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
print(json.dumps(summary, indent=2))
print(coefficients.to_string(index=False))
