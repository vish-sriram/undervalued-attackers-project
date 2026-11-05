# Premier League transfers: summer 2025 and summer 2026

The active dataset includes summer 2025 and summer 2026 only. Winter 2026 has been removed from all output tables and the searchable view. Historical player-performance collection was stopped at the user's request. Scope follows Transfermarkt's summer-page classification, including free-agent signings that can occur after the formal deadline, and Premier League membership in each season. This is not a strict date-range or rolling twelve-month filter.

## Files

- `premier_league_transfers.csv` / `.json`: 339 signing transactions, 319 unique Transfermarkt player IDs. Summer 2025 has 166 transfers; summer 2026 has 173. Includes incoming loans, free transfers and 95 transfers between Premier League clubs. Players can appear in both summers.
- `transfers.html`: searchable table with individual source links.
- `all_arrival_entries.csv`: all 642 arrival entries (527 unique player IDs), preserving 265 loan-return entries and 38 internal promotions. A loan from Manchester City U21 to Coventry is an external signing and is included in the main table. Reserve-team names alone are not a reason to exclude a transfer.
- `summary.json`: counts and field coverage.

## Contract dates and missingness

`old_contract_expiry` is taken from the dated transfer's “Remaining contract duration at [old club]” row, available for 292 of 339 transactions. A free transfer is not sufficient evidence to infer an expiry date.

`new_club_contract_expiry_displayed` is available for 246 transactions. It is the current player-profile expiry displayed alongside the transfer details, populated only when the current club ID and joined date both match the receiving club and transfer date. It does **not** independently establish the original contract signed on that date: subsequent renewals could change it. For loans it may describe the loan end rather than the parent-club contract. Original new-contract expiry dates were not independently reconstructed from contemporaneous announcements.

For transparency, `profile_club_at_scrape`, `profile_contract_expiry_at_scrape`, and `profile_joined_date_at_scrape` retain the actual header context, even when it concerns a different club or later move. Such values are not copied into the new-club expiry column. `new_contract_status` explains each row. Blank dates mean unavailable, not no contract; the original old-club term may also differ from a loan agreement.

Dates use ISO YYYY-MM-DD. Fees and market values are Transfermarkt's displayed EUR amounts, not independently verified payments. Numeric fees distinguish a free transfer (0) from an undisclosed/unspecified amount (blank). Loan fees remain loan fees; the table does not assume a permanent purchase price, add-ons, wages, option exercise or amortisation. Market values and ages are those on the window arrival listing, not current-profile values.

## Sources and method

- Summer 2025: https://www.transfermarkt.com/premier-league/transfers/wettbewerb/GB1/saison_id/2025/s_w/s
- Summer 2026: https://www.transfermarkt.com/premier-league/transfers/wettbewerb/GB1/saison_id/2026/s_w/s

Direct HTTP requests returned empty 202 responses and web fetches were inaccessible/403. The public pages loaded in the browser. All 20 clubs' incoming tables were extracted for each summer (328 summer 2025 entries; 314 summer 2026 entries). Each of the 339 signing-detail pages was read for transfer date, source competition and old-contract duration, alongside the profile header. Internal promotions and rows explicitly labelled “End of loan” are excluded from signing counts but retained in the appendix. Transfermarkt can include scheduled/future loan returns on a summer page; the appendix is a source-page inventory, not a claim that every raw arrival occurred during the stated summer.

Raw browser extractions and read-only DOM extraction functions are preserved under `data/transfermarkt/`. New summer 2025 data is in `raw_summer_2025.json` and `details_summer_2025.json`. Summer 2026 is explicitly selected from earlier raw files, which remain unchanged for provenance and may still contain winter records; they are not the active output dataset. `build_transfer_table.py` rebuilds summer-only output tables without network calls. Individual source URLs, player IDs and transfer IDs are in the signing table. Counts reconcile to source-page totals; each signing-detail record has a transfer date in its window's calendar year and a unique window/transfer-ID pair. Historical football accuracy, fee completeness and original new-contract terms have not been independently audited. As of November 4, 2026.
