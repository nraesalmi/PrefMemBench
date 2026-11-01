## Per method

| method | model | rows | state exact | P | R | recommendations satisfied | clarify | contamination | rows with provenance problems |
|---|---|---|---|---|---|---|---|---|---|
| full_history | qwen3:8b | 36 | n/a | n/a | n/a | 30/36 (83.3%) | 0 ok / 0 unnec / 2 missed | 0 | 15/36 (41.7%) |
| gold | gold | 36 | 36/36 (100.0%) | 100.0% | 100.0% | 34/34 (100.0%) | 2 ok / 0 unnec / 0 missed | 0 | 0/36 (0.0%) |
| latest_wins | qwen3:8b | 36 | 19/36 (52.8%) | 73.6% | 68.1% | 31/36 (86.1%) | 0 ok / 0 unnec / 2 missed | 10 | 16/36 (44.4%) |
| proposed | qwen3:8b | 36 | 24/36 (66.7%) | 86.1% | 83.3% | 33/35 (94.3%) | 0 ok / 1 unnec / 2 missed | 8 | 11/36 (30.6%) |
| recent_only | qwen3:8b | 36 | n/a | n/a | n/a | 32/36 (88.9%) | 0 ok / 0 unnec / 2 missed | 0 | 15/36 (41.7%) |
| retrieval | qwen3:8b | 36 | n/a | n/a | n/a | 30/36 (83.3%) | 0 ok / 0 unnec / 2 missed | 0 | 15/36 (41.7%) |
| rolling_summary | qwen3:8b | 36 | n/a | n/a | n/a | 31/36 (86.1%) | 0 ok / 0 unnec / 2 missed | 0 | 15/36 (41.7%) |

## State vs action (recommendations only)

| method | both | recalled but violated | lucky | both wrong |
|---|---|---|---|---|
| gold | 34 | 0 | 0 | 0 |
| latest_wins | 19 | 0 | 12 | 5 |
| proposed | 24 | 0 | 9 | 2 |

## Matched variants (base vs matched, per family)

| method | family | pairs | state base | state matched | delta | satisfied base | satisfied matched | delta |
|---|---|---|---|---|---|---|---|---|
