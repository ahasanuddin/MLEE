# v3 changes — three swaps on the v2 backbone

These three changes refine the detection core while leaving the rest of the v2
project (Layer-1 classifier, MLflow, dashboard, monitoring, snapshot-date
medallion, Airflow, Docker stack) intact. Each is localized.

## 1. No auto-removal  (`src/llm_verifier.py`, `src/config.py`)
The system no longer emits an automated `REMOVE`. The `REMOVE` decision and the
`llm_remove_above` threshold are gone. `decide()` now anchors on
`match_confidence` and routes the top tier to **HUMAN REVIEW (urgent)** instead.
A human can still choose REMOVE from the dashboard — that is a person's action,
not the system's. Decisions are now only `ALLOW` / `HUMAN REVIEW`, plus a
`priority` (normal/urgent).

## 2. Decoupled scoring  (`src/scoring/composite.py`, `src/config.py`)
The flat weighted composite (`text·0.45 + ocr·0.20 + asr·0.20 + meta·0.15`) is
replaced by two outputs from `score_match()`:
- `match_confidence` = `text_score + 0.15 · metadata_score`, but only once
  `text_score ≥ TEXT_CUTOFF` (0.55). Below the gate, metadata is never applied —
  it **corroborates, never rescues**. The +0.15 cap means metadata can never
  carry a sub-0.70 text match into the urgent band.
- `duplicate_type` ∈ {`same_asset`, `re_authored`, `cross_medium`}, from the
  matched medium config + same-medium feature similarity. Drives lane/priority,
  never the keep/remove decision.
The old `compute_composite_score()` is left in the file but unused.

## 3. Cross-medium 3×3 grid  (`src/matching/text_matching.py`, gold + split + inference)
`combined_text` pooling is replaced by a per-medium grid:
- `utils/data_processing_gold_table.py` now carries `title_text`, `ocr_text`,
  `asr_text` into the feature store (alongside `combined_text`, which Layer 1
  still uses); `split_data.py` carries them into the splits.
- `GridTextMatcher` embeds each medium separately into one shared FAISS index
  and compares the full 3×3 query×seed medium grid, taking the best cell. It
  returns `text_score`, `is_cross_medium`, and `medium_transition`.
- `run_inference.py` wires grid match → same-medium feature score (None for
  title/cross-medium) → `score_match` → `decide`.

Layer 1 is unchanged: it still embeds `combined_text` for harmful/clean.

## Regenerating after these changes
```bash
python generate_data.py            # unchanged
python main.py                     # bronze->silver->gold->split (now carries medium texts)
make embed                         # Layer-1 combined_text embeddings (unchanged)
make train && make register        # Layer-1 classifier (unchanged)
python -m src.run_inference        # rebuilds the grid FAISS index on first run
```
Delete `models/faiss_grid.index` (and the old `models/faiss_seed.index`) to force
a rebuild. Run `python tests/test_v3_logic.py` for the model-free logic checks.

## Verified here (sandbox, no model weights / services)
- `tests/test_v3_logic.py`: decoupled scoring + no-auto-removal — pass.
- Grid matcher on the real `test` split text with a deterministic stub embedder:
  all 20 harmful test videos matched at/above the gate, 12 of them cross-medium.
- Full medallion run (`generate_data.py` → `main.py`) confirms the three medium
  texts reach the splits.

Final end-to-end verification with real MiniLM/Ollama/Postgres/MLflow runs on a
machine with those services (the sandbox can't fetch model weights), same as the
caveat in the v2 README's Known Limitations.
