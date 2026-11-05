"""Evaluate an unpenalized hedonic fee regression on the existing summer split."""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from fit_fee_ridge_outgoings import BASE, BUYER, OUT, ROOT, SELLER


def metrics(actual, prediction):
    return {
        "mae_eur_m": mean_absolute_error(actual, prediction),
        "rmse_eur_m": np.sqrt(np.mean((actual - prediction) ** 2)),
        "r_squared_euro": r2_score(actual, prediction),
        "r_squared_log": r2_score(np.log(actual), np.log(prediction)),
    }


def main():
    output = ROOT / "outputs/fee_model_hedonic"
    output.mkdir(exist_ok=True)
    data = pd.read_csv(OUT / "model_deals.csv")
    features = BASE + [SELLER]
    train = data.loc[data.window.eq("summer_2025")]
    test = data.loc[data.window.eq("summer_2026")].copy()
    assert len(train) == len(test) == 37
    assert set(train.player_id).isdisjoint(test.player_id)
    x = train[features]
    log_fee = np.log(train.transfer_fee_eur / 1e6)
    model = LinearRegression().fit(x, log_fee)
    assert model.rank_ == len(features)
    log_predictions = model.predict(test[features])
    test["hedonic_prediction_eur"] = np.exp(log_predictions) * 1e6
    test["hedonic_error_eur"] = test.hedonic_prediction_eur - test.transfer_fee_eur
    ridge = pd.read_csv(OUT / "holdout_predictions.csv").set_index("transfer_id")
    assert set(ridge.index) == set(test.transfer_id)
    test["ridge_prediction_eur"] = test.transfer_id.map(ridge.seller_dummy_prediction_eur)
    median = train.loc[train[BUYER].eq(1), "transfer_fee_eur"].median() / 1e6
    summary = {
        "target": "natural log of transfer fee in millions of euros",
        "training_log_r_squared": model.score(x, log_fee),
        "training_coefficients": dict(zip(["intercept", *features], [model.intercept_, *model.coef_])),
        "training_inbound_median_fee_eur_m": median,
    }
    for label, rows in {"all": test, "inbound": test.loc[test[BUYER].eq(1)],
                        "outbound": test.loc[test[BUYER].eq(0)]}.items():
        actual = rows.transfer_fee_eur / 1e6
        summary[label] = {
            "deals": len(rows),
            "hedonic_ols": metrics(actual, rows.hedonic_prediction_eur / 1e6),
            "ridge": metrics(actual, rows.ridge_prediction_eur / 1e6),
        }
    inbound = test.loc[test[BUYER].eq(1)]
    summary["inbound"]["training_median_baseline"] = metrics(inbound.transfer_fee_eur / 1e6, median + np.zeros(len(inbound)))
    test.to_csv(output / "holdout_predictions.csv", index=False)
    (output / "metrics_and_coefficients.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))
    print(inbound.sort_values("hedonic_error_eur").loc[:, ["name", "transfer_fee_eur",
          "hedonic_prediction_eur", "ridge_prediction_eur"]].to_string(index=False))


if __name__ == "__main__":
    main()
