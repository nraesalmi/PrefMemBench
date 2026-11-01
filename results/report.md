## Per method

| method | model | rows | state exact | P | R | recommendations satisfied | clarify | contamination | rows with provenance problems |
|---|---|---|---|---|---|---|---|---|---|
| full_history | qwen3:8b | 193 | n/a | n/a | n/a | 152/193 (78.8%) | 0 ok / 0 unnec / 21 missed | 0 | 119/193 (61.7%) |
| gold | gold | 193 | 193/193 (100.0%) | 100.0% | 100.0% | 172/172 (100.0%) | 21 ok / 0 unnec / 0 missed | 0 | 0/193 (0.0%) |
| latest_wins | qwen3:8b | 193 | 88/193 (45.6%) | 79.3% | 66.3% | 138/193 (71.5%) | 0 ok / 0 unnec / 21 missed | 45 | 40/193 (20.7%) |
| proposed | qwen3:8b | 193 | 158/193 (81.9%) | 91.3% | 97.9% | 183/188 (97.3%) | 2 ok / 3 unnec / 19 missed | 38 | 19/193 (9.8%) |
| recent_only | qwen3:8b | 193 | n/a | n/a | n/a | 140/189 (74.1%) | 2 ok / 2 unnec / 19 missed | 0 | 88/193 (45.6%) |
| retrieval | qwen3:8b | 193 | n/a | n/a | n/a | 145/188 (77.1%) | 3 ok / 2 unnec / 18 missed | 0 | 85/193 (44.0%) |
| rolling_summary | qwen3:8b | 193 | n/a | n/a | n/a | 152/191 (79.6%) | 2 ok / 0 unnec / 19 missed | 0 | 119/193 (61.7%) |

## State vs action (recommendations only)

| method | both | recalled but violated | lucky | both wrong |
|---|---|---|---|---|
| gold | 172 | 0 | 0 | 0 |
| latest_wins | 88 | 0 | 50 | 55 |
| proposed | 154 | 0 | 29 | 5 |

## Matched variants (base vs matched, per family)

| method | family | pairs | state base | state matched | delta | satisfied base | satisfied matched | delta |
|---|---|---|---|---|---|---|---|---|
| full_history | ambiguous_then_clarify | 10 | n/a | n/a | n/a | 100.0% | 100.0% | +0.0pp |
| full_history | ambiguous_vs_resolved | 6 | n/a | n/a | n/a | 100.0% | 100.0% | +0.0pp |
| full_history | correction_of_temporary | 15 | n/a | n/a | n/a | 100.0% | 100.0% | +0.0pp |
| full_history | correction_vs_none | 6 | n/a | n/a | n/a | 100.0% | 100.0% | +0.0pp |
| full_history | diet_swap | 5 | n/a | n/a | n/a | 0.0% | 0.0% | +0.0pp |
| full_history | exception_then_restated | 10 | n/a | n/a | n/a | 100.0% | 50.0% | -50.0pp |
| full_history | hypo_vs_real | 6 | n/a | n/a | n/a | 33.3% | 100.0% | +66.7pp |
| full_history | hypothetical_then_real | 10 | n/a | n/a | n/a | 30.0% | 50.0% | +20.0pp |
| full_history | medium_vs_spicy | 5 | n/a | n/a | n/a | 100.0% | 100.0% | +0.0pp |
| full_history | subject_swap | 6 | n/a | n/a | n/a | 100.0% | 33.3% | -66.7pp |
| full_history | temp_vs_perm | 12 | n/a | n/a | n/a | 50.0% | 100.0% | +50.0pp |
| gold | ambiguous_then_clarify | 10 | 100.0% | 100.0% | +0.0pp | n/a | n/a | n/a |
| gold | ambiguous_vs_resolved | 6 | 100.0% | 100.0% | +0.0pp | n/a | n/a | n/a |
| gold | correction_of_temporary | 15 | 100.0% | 100.0% | +0.0pp | 100.0% | 100.0% | +0.0pp |
| gold | correction_vs_none | 6 | 100.0% | 100.0% | +0.0pp | 100.0% | 100.0% | +0.0pp |
| gold | diet_swap | 5 | 100.0% | 100.0% | +0.0pp | 100.0% | 100.0% | +0.0pp |
| gold | exception_then_restated | 10 | 100.0% | 100.0% | +0.0pp | 100.0% | 100.0% | +0.0pp |
| gold | hypo_vs_real | 6 | 100.0% | 100.0% | +0.0pp | 100.0% | 100.0% | +0.0pp |
| gold | hypothetical_then_real | 10 | 100.0% | 100.0% | +0.0pp | 100.0% | 100.0% | +0.0pp |
| gold | medium_vs_spicy | 5 | 100.0% | 100.0% | +0.0pp | 100.0% | 100.0% | +0.0pp |
| gold | subject_swap | 6 | 100.0% | 100.0% | +0.0pp | 100.0% | 100.0% | +0.0pp |
| gold | temp_vs_perm | 12 | 100.0% | 100.0% | +0.0pp | 100.0% | 100.0% | +0.0pp |
| latest_wins | ambiguous_then_clarify | 10 | 0.0% | 0.0% | +0.0pp | 30.0% | 20.0% | -10.0pp |
| latest_wins | ambiguous_vs_resolved | 6 | 0.0% | 0.0% | +0.0pp | 16.7% | 33.3% | +16.7pp |
| latest_wins | correction_of_temporary | 15 | 0.0% | 33.3% | +33.3pp | 33.3% | 73.3% | +40.0pp |
| latest_wins | correction_vs_none | 6 | 100.0% | 100.0% | +0.0pp | 100.0% | 100.0% | +0.0pp |
| latest_wins | diet_swap | 5 | 100.0% | 0.0% | -100.0pp | 100.0% | 100.0% | +0.0pp |
| latest_wins | exception_then_restated | 10 | 0.0% | 0.0% | +0.0pp | 100.0% | 50.0% | -50.0pp |
| latest_wins | hypo_vs_real | 6 | 100.0% | 100.0% | +0.0pp | 100.0% | 100.0% | +0.0pp |
| latest_wins | hypothetical_then_real | 10 | 100.0% | 100.0% | +0.0pp | 100.0% | 100.0% | +0.0pp |
| latest_wins | medium_vs_spicy | 5 | 100.0% | 100.0% | +0.0pp | 100.0% | 100.0% | +0.0pp |
| latest_wins | subject_swap | 6 | 100.0% | 0.0% | -100.0pp | 100.0% | 0.0% | -100.0pp |
| latest_wins | temp_vs_perm | 12 | 0.0% | 100.0% | +100.0pp | 50.0% | 100.0% | +50.0pp |
| proposed | ambiguous_then_clarify | 10 | 50.0% | 100.0% | +50.0pp | 100.0% | 100.0% | +0.0pp |
| proposed | ambiguous_vs_resolved | 6 | 100.0% | 0.0% | -100.0pp | 100.0% | 100.0% | +0.0pp |
| proposed | correction_of_temporary | 15 | 40.0% | 33.3% | -6.7pp | 100.0% | 86.7% | -13.3pp |
| proposed | correction_vs_none | 6 | 83.3% | 100.0% | +16.7pp | 100.0% | 100.0% | +0.0pp |
| proposed | diet_swap | 5 | 100.0% | 100.0% | +0.0pp | 100.0% | 100.0% | +0.0pp |
| proposed | exception_then_restated | 10 | 100.0% | 60.0% | -40.0pp | 100.0% | 70.0% | -30.0pp |
| proposed | hypo_vs_real | 6 | 100.0% | 100.0% | +0.0pp | 100.0% | 100.0% | +0.0pp |
| proposed | hypothetical_then_real | 10 | 100.0% | 100.0% | +0.0pp | 100.0% | 100.0% | +0.0pp |
| proposed | medium_vs_spicy | 5 | 100.0% | 100.0% | +0.0pp | 100.0% | 100.0% | +0.0pp |
| proposed | subject_swap | 6 | 100.0% | 100.0% | +0.0pp | 100.0% | 100.0% | +0.0pp |
| proposed | temp_vs_perm | 12 | 100.0% | 100.0% | +0.0pp | 100.0% | 100.0% | +0.0pp |
| recent_only | ambiguous_then_clarify | 10 | n/a | n/a | n/a | 50.0% | 100.0% | +50.0pp |
| recent_only | ambiguous_vs_resolved | 6 | n/a | n/a | n/a | 100.0% | 100.0% | +0.0pp |
| recent_only | correction_of_temporary | 15 | n/a | n/a | n/a | 71.4% | 100.0% | +28.6pp |
| recent_only | correction_vs_none | 6 | n/a | n/a | n/a | 100.0% | 100.0% | +0.0pp |
| recent_only | diet_swap | 5 | n/a | n/a | n/a | 0.0% | 0.0% | +0.0pp |
| recent_only | exception_then_restated | 10 | n/a | n/a | n/a | 100.0% | 50.0% | -50.0pp |
| recent_only | hypo_vs_real | 6 | n/a | n/a | n/a | 33.3% | 100.0% | +66.7pp |
| recent_only | hypothetical_then_real | 10 | n/a | n/a | n/a | 22.2% | 44.4% | +22.2pp |
| recent_only | medium_vs_spicy | 5 | n/a | n/a | n/a | 100.0% | 100.0% | +0.0pp |
| recent_only | subject_swap | 6 | n/a | n/a | n/a | 100.0% | 33.3% | -66.7pp |
| recent_only | temp_vs_perm | 12 | n/a | n/a | n/a | 50.0% | 100.0% | +50.0pp |
| retrieval | ambiguous_then_clarify | 10 | n/a | n/a | n/a | 100.0% | 100.0% | +0.0pp |
| retrieval | ambiguous_vs_resolved | 6 | n/a | n/a | n/a | 100.0% | 80.0% | -20.0pp |
| retrieval | correction_of_temporary | 15 | n/a | n/a | n/a | 86.7% | 100.0% | +13.3pp |
| retrieval | correction_vs_none | 6 | n/a | n/a | n/a | 100.0% | 80.0% | -20.0pp |
| retrieval | diet_swap | 5 | n/a | n/a | n/a | 0.0% | 0.0% | +0.0pp |
| retrieval | exception_then_restated | 10 | n/a | n/a | n/a | 100.0% | 90.0% | -10.0pp |
| retrieval | hypo_vs_real | 6 | n/a | n/a | n/a | 33.3% | 100.0% | +66.7pp |
| retrieval | hypothetical_then_real | 10 | n/a | n/a | n/a | 22.2% | 44.4% | +22.2pp |
| retrieval | medium_vs_spicy | 5 | n/a | n/a | n/a | 100.0% | 100.0% | +0.0pp |
| retrieval | subject_swap | 6 | n/a | n/a | n/a | 100.0% | 16.7% | -83.3pp |
| retrieval | temp_vs_perm | 12 | n/a | n/a | n/a | 50.0% | 100.0% | +50.0pp |
| rolling_summary | ambiguous_then_clarify | 10 | n/a | n/a | n/a | 100.0% | 100.0% | +0.0pp |
| rolling_summary | ambiguous_vs_resolved | 6 | n/a | n/a | n/a | 100.0% | 100.0% | +0.0pp |
| rolling_summary | correction_of_temporary | 15 | n/a | n/a | n/a | 100.0% | 100.0% | +0.0pp |
| rolling_summary | correction_vs_none | 6 | n/a | n/a | n/a | 100.0% | 100.0% | +0.0pp |
| rolling_summary | diet_swap | 5 | n/a | n/a | n/a | 0.0% | 0.0% | +0.0pp |
| rolling_summary | exception_then_restated | 10 | n/a | n/a | n/a | 100.0% | 50.0% | -50.0pp |
| rolling_summary | hypo_vs_real | 6 | n/a | n/a | n/a | 33.3% | 100.0% | +66.7pp |
| rolling_summary | hypothetical_then_real | 10 | n/a | n/a | n/a | 50.0% | 50.0% | +0.0pp |
| rolling_summary | medium_vs_spicy | 5 | n/a | n/a | n/a | 100.0% | 100.0% | +0.0pp |
| rolling_summary | subject_swap | 6 | n/a | n/a | n/a | 100.0% | 33.3% | -66.7pp |
| rolling_summary | temp_vs_perm | 12 | n/a | n/a | n/a | 50.0% | 100.0% | +50.0pp |
