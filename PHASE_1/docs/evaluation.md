# Local Phase 1 Evaluation

The local scorer evaluates a run directory against public `gold.csv` files
using the Phase 1 column-level matching rule documented in the
[official technical specification](https://dataagent.top/rules).

## Scoring rule

Each CSV column is normalized independently. Its normalized values are sorted
and hashed to create an order-independent signature. Prediction and gold
signatures are matched one-to-one, including duplicate columns.

```text
recall = matched_columns / gold_columns
penalty = penalty_weight * (extra_columns / predicted_columns)
score = max(0, recall - penalty)
```

Column names, column order, and row order are ignored. Row relationships across
different columns are therefore not evaluated. An added, removed, or duplicated
value changes the signature of the entire affected column.

The default penalty weight is `0.1`. It is explicit in the CLI and report so an
experiment always records the value it used.

## Normalization

Before signature generation, the scorer:

- maps common null tokens to an empty string;
- quantizes numeric values to two decimal places using `ROUND_HALF_UP`;
- optionally removes numeric grouping commas;
- normalizes supported date and timestamp formats;
- strips surrounding whitespace and normalizes CRLF inside raw strings.

Normalization behavior is covered by unit tests. Malformed predictions score
zero for the affected task and include a diagnostic status. Invalid or missing
gold data fails the evaluation instead of silently producing a misleading score.

## Usage

From `PHASE_1/`:

```bash
uv run dabench score-run artifacts/runs/<run_id> \
  --gold-dir data/public/output \
  --input-dir data/public/input \
  --verbose
```

The command writes `scores.json` inside the selected run directory by default.
The report contains scorer configuration, per-task results, aggregate metrics,
missing predictions, read errors, and per-difficulty means.

For a partial run, all public tasks without a prediction receive score `0`.
This keeps partial smoke runs from being confused with full benchmark results.
