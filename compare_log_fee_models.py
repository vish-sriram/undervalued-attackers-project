"""Compare log-fee estimators with identical data, features and window split."""
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from fit_fee_hedonic import metrics
from fit_fee_ridge_outgoings import ROOT, BUYER
from fit_fee_xgboost import OUTPUT as DATA_DIR, NUMERIC, LEAGUES

OUTPUT = ROOT / "outputs/fee_model_comparison"
FLAGGED = {"Emmanuel Emegha": "Deal agreed in September 2025 before feature season ended",
           "Donyell Malen": "January option to buy, permanent deal confirmed in May",
           "Dwight McNeil": "Linked exchange with Brennan Johnson"}


def estimator(regressor, numbers):
    encoding = ColumnTransformer([
        ("numeric", "passthrough", numbers),
        ("league", OneHotEncoder(drop="first", min_frequency=3, handle_unknown="infrequent_if_exist",
                                 sparse_output=False), LEAGUES),
    ])
    return TransformedTargetRegressor(
        regressor=Pipeline([("features", encoding), ("model", regressor)]),
        func=np.log, inverse_func=np.exp,
    )


def main():
    OUTPUT.mkdir(exist_ok=True)
    data = pd.read_csv(DATA_DIR / "model_deals.csv")
    train = data[data.window.eq("summer_2025")]
    test = data[data.window.eq("summer_2026")].copy()
    assert (len(train), len(test)) == (29, 28)
    assert set(train.player_id).isdisjoint(test.player_id)
    assert data[NUMERIC + LEAGUES].notna().all().all()
    candidates = {
        "hedonic_ols": (LinearRegression(), {}),
        "ridge": (make_pipeline(StandardScaler(), Ridge()), {"regressor__model__ridge__alpha": [0.1, 1, 10, 100]}),
        "xgboost": (XGBRegressor(tree_method="hist", learning_rate=0.05, min_child_weight=3,
                                 reg_lambda=5, random_state=2026, n_jobs=1),
                    {"regressor__model__max_depth": [1, 2], "regressor__model__n_estimators": [25, 60, 120]}),
    }
    y_train = train.transfer_fee_eur / 1e6
    summary = {"training_deals": len(train), "holdout_deals": len(test), "target": "natural log of fee in EUR millions",
               "selection_metric": "mean absolute euro error in training-only leave-one-out CV",
               "league_encoding": "Categories with fewer than three training deals per side pooled, unseen leagues use that pool",
               "features": NUMERIC + LEAGUES, "training": {}, "holdout": {}}
    for suffix, numeric in [("", NUMERIC), ("_without_xt", [c for c in NUMERIC if c != "xt_per90"])]:
        columns = numeric + LEAGUES
        for name, (regressor, grid) in candidates.items():
            label = name + suffix
            model = estimator(regressor, numeric)
            if grid:
                search = GridSearchCV(model, grid, cv=LeaveOneOut(), scoring="neg_mean_absolute_error")
                search.fit(train[columns], y_train)
                model = search.best_estimator_
                cv_mae = -search.best_score_
                params = search.best_params_
            else:
                cv_mae = -cross_val_score(model, train[columns], y_train, cv=LeaveOneOut(),
                                          scoring="neg_mean_absolute_error").mean()
                model.fit(train[columns], y_train)
                params = {}
            if name == "hedonic_ols":
                enc = model.regressor_["features"]
                assert model.regressor_["model"].rank_ == enc.transform(train[columns]).shape[1]
            test[label] = model.predict(test[columns])
            summary["training"][label] = {"cv_mae_eur_m": cv_mae, "params": params}
    for group in (0, 1):
        test.loc[test[BUYER].eq(group), "median_baseline"] = y_train[train[BUYER].eq(group)].median()
    test["deal_audit_flag"] = test.name.map(FLAGGED)
    for label, rows in {"all": test, "inbound": test[test[BUYER].eq(1)],
                        "outbound": test[test[BUYER].eq(0)],
                        "all_excluding_flagged": test[test.deal_audit_flag.isna()],
                        "inbound_excluding_flagged": test[test[BUYER].eq(1) & test.deal_audit_flag.isna()]}.items():
        summary["holdout"][label] = {"deals": len(rows)}
        for name in [*summary["training"], "median_baseline"]:
            summary["holdout"][label][name] = metrics(rows.transfer_fee_eur / 1e6, rows[name])
    assert test[list(summary["training"])].gt(0).all().all()
    test.to_csv(OUTPUT / "holdout_predictions_eur_m.csv", index=False)
    (OUTPUT / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for label, scores in summary["holdout"].items():
        print(label, scores["deals"])
        print(pd.DataFrame({k: v for k, v in scores.items() if k != "deals"}).T.round(3).to_string())


if __name__ == "__main__":
    main()
