"""Refit the season-totals fee model on deals with a Premier League buyer."""

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from predict_transfer_fee import MODEL

OUT = MODEL.parent
FEATURES = [
    "age_at_transfer",
    "years_until_selling_contract_expiry",
    "ln_prior_season_minutes",
    "total_xg",
    "total_xt_added",
    "is_winger",
    "tournament_appearances",
]
LEAGUES = ["selling_club_league", "buying_club_league"]


def fit(train):
    encoding = ColumnTransformer(
        [
            ("numeric", "passthrough", FEATURES),
            (
                "league",
                OneHotEncoder(
                    drop="first",
                    min_frequency=3,
                    handle_unknown="infrequent_if_exist",
                    sparse_output=False,
                ),
                LEAGUES,
            ),
        ]
    )
    regressor = Pipeline(
        [("features", encoding), ("model", make_pipeline(StandardScaler(), Ridge()))]
    )
    model = TransformedTargetRegressor(
        regressor=regressor, func=np.log, inverse_func=np.exp
    )
    search = GridSearchCV(
        model,
        {"regressor__model__ridge__alpha": [0.1, 1, 10, 100]},
        cv=LeaveOneOut(),
        scoring="neg_mean_absolute_error",
    )
    search.fit(train[FEATURES + LEAGUES], train.transfer_fee_eur / 1e6)
    return search.best_estimator_, search.best_params_["regressor__model__ridge__alpha"]


def refit(output=MODEL):
    """Fit the bundled 2025 training deals and save the model."""
    deals = pd.read_csv(OUT / "model_deals.csv")
    train = deals[deals.window.eq("summer_2025")]
    model, _ = fit(train)
    dump(model, output)
    return model
