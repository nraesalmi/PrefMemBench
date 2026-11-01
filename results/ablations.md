Extraction: 120 trajectories loaded, 0 extracted, 0 failed turn(s).

### Resolver ablations

| ablation | checkpoints | state exact | P | R | satisfied | empty profile | clarify ok | handled |
|---|---|---|---|---|---|---|---|---|
| all rules on | 193 | 158/193 (81.9%) | 91.3% | 97.9% | 172/178 | 0 | 9 | 181/193 |
| no_scope | 193 | 150/193 (77.7%) | 89.2% | 97.9% | 164/178 | 0 | 9 | 173/193 |
| no_time | 193 | 149/193 (77.2%) | 89.0% | 97.9% | 164/178 | 0 | 9 | 173/193 |
| no_subject | 193 | 136/193 (70.5%) | 85.6% | 97.9% | 166/178 | 0 | 9 | 175/193 |
| no_supersede | 193 | 116/193 (60.1%) | 80.1% | 99.5% | 175/178 | 0 | 9 | 184/193 |
| no_correction | 193 | 155/193 (80.3%) | 90.5% | 97.9% | 172/178 | 0 | 9 | 181/193 |

### Where each rule bites

| ablation | family | n | exact (all on -> off) | handled (all on -> off) |
|---|---|---|---|---|
| no_scope | exception_then_restated | 20 | 16/20 -> 14/20 | 17/20 -> 15/20 |
| no_scope | temp_vs_perm | 24 | 24/24 -> 18/24 | 24/24 -> 18/24 |
| no_time | correction_of_temporary | 30 | 11/30 -> 10/30 | 27/30 -> 27/30 |
| no_time | exception_then_restated | 20 | 16/20 -> 14/20 | 17/20 -> 15/20 |
| no_time | temp_vs_perm | 24 | 24/24 -> 18/24 | 24/24 -> 18/24 |
| no_subject | diet_swap | 15 | 15/15 -> 5/15 | 15/15 -> 15/15 |
| no_subject | subject_swap | 18 | 18/18 -> 6/18 | 18/18 -> 12/18 |
| no_supersede | correction_of_temporary | 30 | 11/30 -> 11/30 | 27/30 -> 30/30 |
| no_supersede | correction_vs_none | 12 | 11/12 -> 8/12 | 11/12 -> 11/12 |
| no_supersede | hypo_vs_real | 12 | 12/12 -> 6/12 | 10/12 -> 10/12 |
| no_supersede | hypothetical_then_real | 20 | 20/20 -> 15/20 | 20/20 -> 20/20 |
| no_supersede | medium_vs_spicy | 10 | 10/10 -> 0/10 | 10/10 -> 10/10 |
| no_supersede | subject_swap | 18 | 18/18 -> 12/18 | 18/18 -> 18/18 |
| no_supersede | temp_vs_perm | 24 | 24/24 -> 12/24 | 24/24 -> 24/24 |
| no_correction | correction_of_temporary | 30 | 11/30 -> 10/30 | 27/30 -> 27/30 |
| no_correction | correction_vs_none | 12 | 11/12 -> 9/12 | 11/12 -> 11/12 |
