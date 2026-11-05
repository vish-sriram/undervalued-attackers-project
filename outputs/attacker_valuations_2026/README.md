# Attacker model estimates

Valuation snapshot: 5 November 2026. Hypothetical permanent move to a Premier League buyer. For loans, the selling club is the parent club; loan options are not priced.

The retained model exponentiates predicted log fees. These are model point estimates, not calibrated arithmetic expected values, bids, or independently verified market values. No retraining or retransformation correction was applied.

19 training deals; 14 holdout deals; holdout euro R² 0.697 and MAE €13.79m. Performance on this small holdout does not establish accuracy for these players. Current injury, loan options, and release clauses are not model inputs.

45,324 raw events; 43,800 converted actions; 27,021 scored successful move actions. Shots and other ineligible actions have blank xT, not zero. Amad has one eligible action with missing endpoint coordinates; no endpoint was invented.

The 34-player screen uses the available 2025/26 cache, with partial French and German league coverage. Player minutes are cross-checked against FotMob within max(20 minutes, 2%). Identities use league/name matching and age checks, not independent proof for every player. Raw events and converted actions are separate files because the conversion can infer carries and omit non-action events.

| Player | Model fee (€m) | Status / extrapolation |
|---|---:|---|
| Bukayo Saka | 597.25 | input_checks_passed; outside training range: tournament_appearances |
| Erling Haaland | 255.97 | input_checks_passed; outside training range: years_until_selling_contract_expiry; total_xg |
| Bradley Barcola | 193.52 | input_checks_passed; outside training range: tournament_appearances |
| Pedro Neto | 149.02 | input_checks_passed; outside training range: years_until_selling_contract_expiry; tournament_appearances |
| Antoine Semenyo | 124.36 | input_checks_passed; outside training range: tournament_appearances |
| João Pedro | 119.48 | input_checks_passed; outside training range: years_until_selling_contract_expiry |
| Iliman Ndiaye | 108.43 | input_checks_passed; outside training range: tournament_appearances |
| Igor Thiago | 105.03 | input_checks_passed; outside training range: total_xg |
| Benjamin Sesko | 88.87 | input_checks_passed |
| Brian Brobbey | 82.75 | input_checks_passed; outside training range: total_xt_added |
| Brenden Aaronson | 74.34 | input_checks_passed; outside training range: tournament_appearances |
| Nick Woltemade | 70.74 | input_checks_passed |
| Hugo Ekitiké | 63.91 | input_checks_passed |
| Jørgen Strand Larsen | 63.51 | input_checks_passed |
| Noah Okafor | 58.89 | input_checks_passed |
| Mohammed Kudus | 54.77 | input_checks_passed |
| Anthony Gordon | 54.00 | input_checks_passed; outside training range: tournament_appearances |
| Dango Ouattara | 48.95 | input_checks_passed |
| Thierno Barry | 47.41 | input_checks_passed |
| Pavel Sulc | 47.21 | input_checks_passed |
| Igor Jesus | 47.10 | input_checks_passed |
| Joaquín Panichelli | 45.68 | input_checks_passed |
| Jaidon Anthony | 44.77 | input_checks_passed |
| Kevin Schade | 42.84 | input_checks_passed |
| Omari Hutchinson | 41.28 | input_checks_passed |
| Martial Godo | 31.61 | input_checks_passed |
| Tim Lemperle | 31.02 | input_checks_passed; seller league unseen in training |
| Callum Hudson-Odoi | 28.99 | input_checks_passed |
| Igor Paixão | 28.34 | input_checks_passed |
| Félix Correia | 25.92 | input_checks_passed |
| Mason Greenwood | 25.85 | input_checks_passed; seller league unseen in training |
| Fisnik Asllani | 25.04 | input_checks_passed; seller league unseen in training |
| Amad Diallo | — | Eligible xT actions missing scores |
| Andrej Ilic | — | Current contract expiry unavailable or expired; seller league unseen in training |

Source-specific corrections and URLs are retained in source_overrides.json and the valuation table. Woltemade’s missing Stuttgart xG is resolved from a match shot map with no shots attributed to him, as recorded in woltemade_stuttgart_xg_verification.json. Ilić remains unpriced: his club confirms an extension without an expiry date, and third-party reports do not securely establish the new term.
