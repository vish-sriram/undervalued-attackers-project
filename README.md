# Undervalued attackers

A transfer-fee research project using WhoScored events, FotMob league and international statistics, and Transfermarkt deal records. The retained estimator is a ridge regression on log fees, fitted on permanent fee-paying transfers to Premier League buyers. It uses season-total xG and xT, minutes, age, selling-contract years, winger status, senior tournament appearances, and selling league. A Premier League buyer is assumed.

## Predict or refit

Use Python 3.12. Install the model dependencies, then run the offline checks.

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python verify_project.py
.venv/bin/python predict_transfer_fee.py players.csv predictions.csv
```

The input columns are documented in `predict_transfer_fee.py --help`. The output appends `predicted_fee_eur`. The model can be refitted with `.venv/bin/python fit_fee_arrivals_only.py` from the included 33 deal rows in [model_deals.csv](outputs/fee_model_international/arrivals_only/model_deals.csv). The serialized artifact loads with Joblib and should only be loaded from a trusted checkout.

## Results and limits

- The retained model trained on 19 summer 2025 arrivals and was tested on 14 summer 2026 arrivals. Its euro R² was 0.697 and MAE was €13.79m. Log-fee R² was 0.397. This holdout was reused during model selection. [Metrics](outputs/fee_model_international/arrivals_only/metrics.json)
- The event shortlist contains 34 players, 32 with priced scenarios. The final club and recent-move screen leaves seven players, six with estimates. The underlying French and German event caches cover part of each league. [Shortlist audit](outputs/attacker_valuations_2026/README.md)
- Predictions exponentiate log-fee estimates. They are model point estimates for a hypothetical Premier League move. Extrapolation, negotiated fee structures, and source gaps can materially change interpretation.
- xT scores successful possession-retaining moves using the saved xT surface. Raw event counts and converted actions describe different levels of the pipeline. Reproducibility and source reconciliation do not establish independent video accuracy.
- The [Arsenal output comparison](outputs/attacker_valuations_2026/arsenal_output_comparison.csv) reports both season-total and per-90 differences. Tzolis's [Club Brugge xG](https://www.fotmob.com/en-GB/leagues/40/stats/season/27152/players/expected_goals/team/8342/club-bruggeplayersplayers) is available, but the local event cache cannot supply a comparable Belgian league xT total.

## Data boundary

The repository includes code, the retained model, compact model inputs, derived summaries, and audit results. Raw provider caches, match-level exports, working databases, and the local Python environment are excluded from Git. They remain local and are required for replaying the full collection and event derivation. The historical collection scripts additionally use SoccerData, Socceraction, DuckDB, Requests, Beautiful Soup, and XGBoost for historical experiments.
