# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Mansi Negi
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/MansiNegi281/flyrank_ml_internship
- **Date:** [15th August 2026]

## 1. Problem framing
Unit of analysis: one content page (`content_id`), aggregated over a 90-day
window with a 30-day trend comparison. Output: a ranked queue of pages by
predicted decline probability, with a reason code. Action: a content team
lead reviews the top of the queue first each cycle instead of reviewing
pages at random. Cost of a wrong call: reviewing pages that were already
fine wastes editor time; missing genuinely declining pages lets traffic
loss compound before anyone notices. ML helps because decline isn't
explained by any single signal — position, CTR, staleness, and volume
interact, which a fixed rule can't weigh as well as a model trained on the
real relationships in the data.

## 2. Data safety
Data: FlyRank internship warehouse (`dim_content`, `fact_content_daily_performance`),
401,259 content pages aggregated over a 90-day trailing window. `client_hash_id`
and `content_hash_id` are used only for grouping the validation split and
identifying rows — never as model features. Leakage check: correlation
between every feature and the label was computed; highest was 0.351
(`avg_position`), well under a 0.9 leakage threshold. Neither
`trend_direction` nor the raw impression values it's derived from are used
as features. No client names, domains, or private queries appear anywhere
in this repo's `work/` folder.

## 3. Baseline
A hand-written weighted score (Week 4) combining search volume, trend
magnitude, staleness, and position, reporting precision 0.5723. **Caveat:**
this baseline was computed on the smaller anonymized starter CSV (30K
rows), while the model below runs on the full warehouse (401K rows) — the
two numbers are directionally comparable but not a strictly controlled
apples-to-apples comparison. A more rigorous version of this report would
recompute the baseline rule directly on the warehouse.

## 4. Model / analysis
Method: Random Forest classifier (100 trees, `random_state=42`), with
Logistic Regression as a secondary comparison. Random Forest fits this lane
because decline risk depends on non-linear interactions between features
(e.g. high volume + poor position matters more together than either alone).

Feature list: `search_volume`, `ctr`, `avg_position`, `engagement_rate`,
`scroll_rate`, `content_age_days`, `days_since_last_update`,
`impressions_90d`, `sessions_90d`.

Left out on purpose: `client_id`/`content_id` (identifiers, not signals).

Target: `is_declining` = 1 if 30-day impressions fell more than 10% versus
the prior 30 days, else 0 — an observed-rule proxy, not a ground-truth label.

## 5. Evaluation
Split: client-grouped (not random) — no client's pages appear in both train
and test, motivated by a concrete finding in the Week 4 baseline queue where
5 of the top 5 highest-scored pages belonged to a single client.

**Base rate:** 44.5% of pages are labeled declining (majority class 55.5%).
A trivial always-predict-declining classifier would score ~44.5% precision
at 100% recall — the relevant floor to judge the model against.

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Naive always-positive | ~0.445 | 1.0 | — |
| Baseline hand rule (anonymized CSV) | 0.5723 | — | — |
| Random Forest (grouped split, full warehouse) | 0.5835 | 0.7738 | 0.6653 |

The model beats the naive floor by roughly 14 points of precision while
keeping recall high, meaning it catches most true decliners without
flooding the queue with false positives as badly as guessing would.

Error analysis: the model likely over-flags pages with one strong signal in
isolation (e.g. high search volume alone on an otherwise healthy page) and
is most confident on pages with converging negative signals.

## 6. Interpretation
Correlation with the label (proxy for influence, since full feature
importances weren't extracted in this run): `avg_position` (0.351) is by
far the strongest single signal, followed by `content_age_days` (-0.076,
older pages slightly less likely to be flagged as currently declining) and
`impressions_90d` (0.076). CTR, engagement, and scroll rate were weak
standalone signals (all under 0.04 correlation) — a negative result worth
stating plainly rather than overselling engagement metrics as strong
independent predictors of decline.

## 7. Recommendation
1. Review the top of the ranked queue first each cycle — highest predicted
   decline probability
2. Treat reason codes (`DECLINING_TRAFFIC`, `POOR_POSITION`,
   `STALE_CONTENT`, `LOW_CTR`) as a starting hypothesis, not a diagnosis
3. Never fully automate publishing or de-indexing decisions from this queue
4. Re-evaluate if precision on fresh data drops meaningfully below 0.58

Confidence: this is an observed, decision-support signal on this dataset —
not a causal claim, and not a statement about how Google's algorithm works.

## 8. Reproducibility
```bash
git clone https://github.com/MansiNegi281/flyrank_ml_internship.git
cd flyrank_ml_internship
pip install -r requirements.txt
```
Then open `work/notebooks/capstone.ipynb` in Colab and run all cells top to
bottom. Random seed fixed at `random_state=42` (train/test split and Random
Forest). Data source: `hf://datasets/FlyRank/internship-warehouse`.

## Acknowledgments & Data Credit
Built on the FlyRank ML Internship dataset — [flyrank.ai](https://flyrank.ai)
