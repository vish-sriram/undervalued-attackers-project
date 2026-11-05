-- Team and player events, derived from the dumped SPADL match stream.
-- Each team's actions are ordered as they occurred, then each player's.

CREATE OR REPLACE TABLE team_events AS
SELECT *
FROM match_events
WHERE team_id IS NOT NULL
ORDER BY team_id, game_id, period_id, time_seconds, action_id;

CREATE OR REPLACE TABLE player_events AS
SELECT *
FROM match_events
WHERE player_id IS NOT NULL
ORDER BY player_id, game_id, period_id, time_seconds, action_id;
