# Undervalued attackers

Transfer-fee estimates for Premier League buyers. The saved model fits log-ridge to 19 summer 2025 deals. On 14 summer 2026 deals it had an R² of 0.697 and mean absolute error of €13.79m, both scored in euros. We reused that holdout during model selection. [Model results](outputs/fee_model_international/arrivals_only/metrics.json)

## Use the model

Install `requirements.txt` with Python 3.12. Use the saved model from Python:

```python
import pandas as pd
from predict_transfer_fee import predict_fees

players = pd.read_csv("players.csv")
predict_fees(players).to_csv("predictions.csv", index=False)
```

Inputs are age at transfer, years remaining on the seller's contract, prior-season league minutes, total xG, total xT added, a winger flag, senior World Cup/Copa América/Euro appearances, and the seller's league. The exact column names are in `predict_transfer_fee.py`. Predictions are in euros. Use the Joblib file only from a trusted checkout.

`fit_fee_arrivals_only.refit()` rebuilds the saved model from the bundled training deals. `verify_project.verify()` checks the refit, holdout predictions, and shortlist estimates. `expected_threat.read_match_actions()` scores cached WhoScored match files with the saved xT surface and requires Socceraction 1.5.3.

## Data

- [Top-five league transfers](outputs/top_five_transfer_fees/top_five_transfer_fees_2022_2026.csv) covers winter 2022 through summer 2026. It contains 11,224 unique deals. The [sterling dataset](outputs/top_five_transfer_fees/permanent_transfers_converted_to_gbp_2022_2026.csv) converts the 1,146 reported permanent deals above €10m. The [£10m subset](outputs/top_five_transfer_fees/permanent_transfers_above_10m_gbp_2022_2026.csv) applies the threshold to nominal pounds.
- Sterling fees use the ECB's average daily pounds-per-euro rate for January or June–August because the transfer listings give windows. Real fees multiply nominal sterling by September 2026 UK CPI divided by the average CPI for those months. [ECB series](https://data.ecb.europa.eu/data/datasets/EXR/EXR.D.GBP.EUR.SP00.A) · [ONS CPI](https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/d7bt/mm23)
- The [scouting shortlist](outputs/attacker_valuations_2026/README.md) has 34 players and 32 priced scenarios. The French and German match caches are partial. The [Arsenal comparison](outputs/attacker_valuations_2026/arsenal_output_comparison.csv) preserves season and per-90 differences.

Raw provider caches and local databases stay out of Git. Earlier experiments and collection scripts are available in commit `bbd7d3b`.
