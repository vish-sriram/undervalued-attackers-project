# Pre-transfer attacker data: collection and modelling notes

## Scope and current status

This extraction targets the 71 unique attackers whose latest qualifying summer 2025 or summer 2026 arrival came from a top-five league or Ajax. For players transferred in both summers, the latest transfer is the outcome. This preserves the previously reported 71-player pool; it is not 71 independent permanent purchases.

Use `run_status.json` for the current pipeline state and `summary.json` for the last built counts. While the state is `collecting`, the CSV files and SQL tables are provisional. A launched scraper or an observed season label does not establish complete season coverage.

Performance windows are **one completed European league season before each transfer**:

- Summer 2025: **2024/25 only**.
- Summer 2026: **2025/26 only**.

No season is silently substituted when one is missing. Older raw downloads are retained as cache/provenance, but 2023/24 is excluded from all active performance tables. Calendar-year leagues, youth competitions and uncovered previous clubs may require additional sources; gaps remain explicit. The selling club's league is not assumed to be the player's performance league: loans and earlier teams are identified using historical season records. FotMob namesakes are resolved by exact date-of-birth matches against Transfermarkt profiles, with source-specific IDs and an auditable crosswalk. WhoScored historical responses expose current age, so that field is used only for an identity check against current date of birth, not as a historical model feature.

## Outputs

- `player_transfer_features.csv`: one row per player with available event data from the single requested prior season. Includes coverage flags and risk measures.
- `player_season_features.csv`: the same event metrics separately for each observed season.
- `player_match_metrics.csv`: the underlying WhoScored match observations.
- `fotmob_player_league_seasons.csv`: independently sourced historical xG and minutes, with source URLs.
- `fotmob_player_match_xg.csv`: match-level xG observations used for uncertainty estimates.
- `paired_match_risk_inputs.csv`: xG and xT aligned to the same player, league, date and team, preserving their covariance for joint match resampling.
- `coverage_audit.csv`: all 71 targets and their requested versus observed seasons. This is an audit, not invented performance records for unavailable players.
- `model_ready_features.csv`: only positive-fee permanent deals with the requested prior season, completed required-source collection, reconciled minutes, and the requested fields present. This may be empty while collection runs.
- `extraction_errors.csv`: processing failures or rejected identities. Empty output means no processing failures in the last build, not proof of football accuracy or complete source coverage.

The SQL database is `attacker_transfer_history.duckdb` in the project root. Tables are `attacker_transfer_metadata`, `attacker_player_match_stats`, `attacker_player_season_features`, `attacker_transfer_features`, `attacker_coverage_audit`, `attacker_xg_source_seasons`, `attacker_xg_source_matches`, and `attacker_paired_match_risk_inputs`. The existing Premier League 2025/26 database is preserved.

## Column definitions and sources

| Requested field | Definition and qualification |
|---|---|
| Name | Transfermarkt cohort name; source IDs retained for joining. |
| Total minutes | Reconstructed from WhoScored lineups, substitutions and dismissals, capped to regulation time. Fractional substitution minutes use seconds where supplied; added-time inflation is excluded. These estimates can differ slightly from published provider minutes. |
| Total xT added | Signed net sum of successful passing/carrying move values from the common xT surface. Backward moves can contribute negatively. `total_positive_xt_added` is also retained as a different measure. |
| Total receptions | Count of observed on-ball actor changes, including gains from opponents, using raw WhoScored `isTouch` events and resetting at period/restart boundaries. This implements an event-sequence proxy for the requested definition; it is not verified continuous possession tracking or a complete pass-recipient feed. Repeated touches by one holder do not create additional entries, and synthetic SPADL carries are not counted as receptions. |
| Shots per 90 | WhoScored raw shot events, excluding own goals, divided by reconstructed minutes and multiplied by 90. |
| xG per 90 | Separate FotMob xG source. The season field uses published rounded xG totals divided by FotMob's own minutes, so provider denominators are not mixed. Match shotmaps retain more precise xG and non-penalty xG for risk calculations. xG includes penalties unless a field explicitly says non-penalty. Unavailable xG is not estimated from shots. |
| Transfer fee | Transfermarkt displayed fee converted to EUR numeric units. Unknown amounts remain missing; genuine free transfers are zero. Loans are labelled separately and excluded from the permanent-fee modelling subset. Fees/add-ons are not independently verified. |
| Selling-club league | Transfermarkt's competition on the dated transfer detail page. |
| Injury missed league matches (%) | 100 × distinct team league fixtures overlapping reported Transfermarkt injury dates, excluding observed appearances, divided by the relevant full team-season league fixture count. This is a reported-date overlap estimate, not a verified medical absence log. Illness, suspension, fitness and personal-absence entries are excluded. No entries available means unknown, not zero. Multiple teams within one season require registration dates before a denominator is supplied. |
| Years to old contract expiry | Non-negative days between the transfer date and Transfermarkt's historical old-contract expiry / 365.25. New/current profile expiry is not used. Unknown dates remain missing. Parent-club employment contracts and loan agreements must not be conflated. |
| Position | Transfermarkt position at the transfer listing. |

Transfermarkt's displayed injury games-missed totals can include cup games. They are therefore not directly divided by league-only fixture counts. The raw episode tables are retained in `data/player_history/` for review. Zero reported overlap is not a medical guarantee of no injury.

Understat was attempted through SoccerData but returned a maintenance HTML page (“We will come back soon”) instead of its API JSON. FotMob is the separately attributed fallback, as authorized. Raw pages, linked statistics JSON, match JSON and collection status are cached.

## Time cutoffs and xT validity

Every performance match is checked against its player's transfer date and requested seasons. A common 2024/25 Premier League xT surface is fitted only once all 380 raw training games are present, with the away-team coordinate mirror undone as in the existing xT script. This surface predates both transfer summers. Until that surface is ready, the existing 2025/26 model can score summer 2026 rows only; earlier transfer rows have missing xT rather than leaked future-trained values. Missing eligible-action coordinates are flagged rather than valued at zero.

## Risk and uncertainty

Both match-to-match variation and uncertainty in the mean are retained:

- `xt_weighted_match_sd_per90` and `xg_weighted_match_sd_per90`: minute-weighted dispersion of match rates.
- `xt_se_per90` and `xg_se_per90`: cluster-by-match standard errors of the total-value / total-minutes ratio. At least five appearances and 180 minutes are required to report these estimates; those thresholds are explicit safeguards, not established universal cutoffs.
- Effective match counts, observed minutes and match counts identify unequal exposure. A one-match player must not receive a false zero-risk score merely because a standard deviation cannot be estimated.
- `minutes_reliability_900 = minutes / (minutes + 900)` and the one-standard-error xT lower scenario are transparent illustrative settings, not calibrated probabilities or validated bid adjustments.
- There is no between-season SD in the one-season dataset. Volatility is estimated from matches within that season.

`risk_adjusted_rate` in the metrics module supports a transparent scenario:

1. Shrink a player's rate toward a position-specific training-set prior, with less shrinkage as minutes increase.
2. Subtract an uncertainty penalty proportional to the standard error.
3. Optionally subtract a separate volatility penalty proportional to match SD.

The prior, exposure scale and penalty weights must be estimated/tuned using training data only. Coefficient of variation is avoided because net xT can be close to zero or negative. Missing uncertainty stays missing.

No euro risk penalty or fee regression has been fitted in this extraction. For the fee model, propagate resampled match-level performance through the fitted model to obtain a fee distribution, then validate a conservative bid quantile on held-out transfers. Injury can affect both expected price and risk; avoid applying the same injury discount twice. A predicted market fee is not, by itself, the player's sporting value or a recommended bid.

## Verification

The metric tests cover actor-change counting, unused substitutes, dismissals, exposure-aware uncertainty and conservative scenarios. The final pipeline check verifies unique identities, requested seasons, pre-transfer cutoffs, plausible minute ranges, aggregate reconciliation, injury percentage bounds and permanent-fee eligibility. These checks establish reproducibility and internal consistency, not independent annotation accuracy or complete public-source coverage.
