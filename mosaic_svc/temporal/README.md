# Mosaic Temporal Style Adapter P1

P1 consumes a Temporal Timbre Memory query path and applies the retrieved local target style to frozen Seed-VC inference.

## Injection point

Seed-VC repeats one 192-dimensional CAMPPlus style vector across every acoustic frame before `cond_x_merge_linear`. P1 wraps that merge module and replaces only the style slice with a time-varying schedule.

The original DiT, Flow Matching solver, Content condition, F0 condition, prompt mel, and vocoder are unchanged.

```text
TTM query at 10 Hz
  -> candidate patch context CAMPPlus embeddings
  -> soft top-k average
  -> canonical-relative norm clipping
  -> confidence gate
  -> temporal smoothing
  -> interpolation to Seed-VC mel frames
  -> style slice replacement
```

The CFG null branch keeps its zero style and is never replaced by the target schedule.

## Safe defaults

```text
max_gate:              0.25
max_norm_ratio:        0.10
min_confidence:        0.45
min_source_f0_conf:    0.35
min_patch_f0_conf:     0.50
min_patch_quality:     0.90
min_weight_margin:     0.015
max_register_distance: 0.20
max_voiced_ratio_diff: 0.35
smoothing_seconds:     0.50
patch_context_seconds: 2.0
```

The maximum effective deviation from canonical style is approximately `0.25 * 0.10 = 2.5%` of the canonical vector norm. Frames below the confidence threshold, with ambiguous top candidates, low F0 confidence, mismatched voicing/register, or low-quality target patches use canonical style exactly. Rejection reasons are counted in the sibling `.temporal.json` report.

## Inference

```powershell
python -m mosaic_svc.p0.infer_p0 `
  --source source_vocal.wav `
  --prompt canonical_prompt.wav `
  --output out\ttm1 `
  --diffusion-steps 60 `
  --inference-cfg-rate 0.50 `
  --f0-condition True `
  --temporal-query out\temporal_query.jsonl `
  --temporal-memory out\temporal_memory
```

Each generated WAV receives a sibling `.temporal.json` report containing active frame count, unique patches, gate statistics, style-change ratios, and all runtime settings.

## Diagnostic probe

The following settings are intentionally stronger and should only be used to determine whether the conditioning direction is useful:

```text
--temporal-max-gate 0.60
--temporal-max-norm-ratio 0.20
--temporal-smoothing-seconds 0.70
```

This permits roughly 12% maximum and approximately 7% mean style deviation in the first test. It is not the production default.

## Known limitations

- Local values are inferred with CAMPPlus from a context window around each retrieved patch.
- CAMPPlus was trained for global speaker identity, not frame-level singing timbre.
- P1 is inference-only and has no learned projection from TTM spectral values to Seed-VC style space.
- Query updates occur at 10 Hz and are interpolated to mel frames.
- The adapter may improve local identity, do nothing, or introduce timbre wobble. Listening evaluation is required.
