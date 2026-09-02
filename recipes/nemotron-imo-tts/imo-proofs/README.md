# IMO 2026 submissions

`imo2026_submissions.jsonl` holds the six proofs this pipeline submitted at IMO 2026 (rows `P1` to `P6`, 30 of 42
points in total) and the later Problem 6 proof from the continued run described in [the report](../paper.pdf), Section 4.3
(row `P6-continued`, produced past the contest cutoff). Each row has `problem_idx`, `problem` (the statement),
`proof` (the text exactly as the model produced it), and `official_score`; the continued-run row carries
`unofficial_score` instead, the 4 of 7 awarded by independent graders, which is not an official result.
