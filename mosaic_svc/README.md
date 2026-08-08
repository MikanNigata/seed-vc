# Mosaic-SVC Seed-Only Extension Notes

## Temporal Timbre Memory P1

TTM-P1 can replace Seed-VC's repeated global style slice with a bounded, confidence-gated per-frame schedule derived from a TTM query. Content, F0, prompt mel, DiT, and vocoder remain frozen. See [the P1 design and usage guide](temporal/README.md).

This path is experimental. Safe defaults limit the effective canonical-style deviation to approximately 2.5%; stronger settings are diagnostic only.

---

This fork keeps upstream Seed-VC intact and adds Mosaic-SVC as an extension package.

## Active Scope

Only the frozen Seed-VC P0-P10 path is active. HQ-SVC is not part of this repository, and Mosaic-SVC R1.6 P11-P16, Streaming Student, AP Head, NSF, Refiner, live runtime, and GUI were retired after unacceptable subjective audio quality. Their historical source remains for reproducibility, but every training and inference entrypoint rejects execution.

## Implemented

- P0 frozen Seed-VC inference with fixed canonical prompt.
- P0 prompt/style ablation via separate `--prompt` and `--style-audio`.
- P0 inference-only CAMPPlus prototype correction.
- P0 Style-Slice Adapter rank 4, installed by wrapping `cond_x_merge_linear`.
- P0 Style-Slice Adapter training that updates only the adapter.
- M1 prompt candidate sweep and prompt/style path ablation.
- M2 Prompt Adapter, installed by wrapping `cond_x_merge_linear` with a prompt mel/condition/style residual branch.
- M3 Prompt Adapter training over frozen Seed-VC using high-quality singing clips only.
- M4 dialogue CAMPPlus speaker profile extraction and high-quality prompt reranking.
P10 is the current default. Do not resume or extend the retired R1.6 path.

## M1-M4 Current Result

The dialogue profile is built from the cleaned 25-minute dialogue file, but dialogue audio is not used as an acoustic teacher.

```text
D:\voice-lab\out\mosaic_svc\speaker_profiles\maneki_dialogue25_campplus.pt
D:\voice-lab\out\mosaic_svc\speaker_profiles\maneki_dialogue25_campplus.csv
```

Prompt reranking by dialogue speaker profile ranked the 12-second Dadadada prompt candidates as:

```text
01 prompt_05_048.00s sim=0.425142 score=0.594224
02 prompt_07_072.00s sim=0.423787 score=0.584651
03 prompt_08_084.00s sim=0.387137 score=0.554121
04 prompt_03_024.00s sim=0.483816 score=0.545296
05 prompt_06_060.00s sim=0.346047 score=0.529108
```

Initial M2/M3 evaluation used P05 as canonical because it was the top dialogue-profile rerank candidate.

```text
D:\voice-lab\out\mosaic_svc\p0\m1_m2_compare_p05_cfg050_steps60\M1_P05_raw_lufs.wav
D:\voice-lab\out\mosaic_svc\p0\m1_m2_compare_p05_cfg050_steps60\M2_P05_prompt_adapter_lufs.wav
D:\voice-lab\out\mosaic_svc\p0\m1_m2_compare_p05_cfg050_steps60\metrics.csv
```

Metrics from the first 300-step Prompt Adapter run:

```text
M1 raw: f0_corr=0.992921 cent_rmse=54.45 uv_mismatch=0.136997
M2 adapter: f0_corr=0.994056 cent_rmse=50.68 uv_mismatch=0.147833
```

This is not a production-quality improvement yet. It is a controlled first check that prompt-path adaptation can move the model without immediately damaging F0.

Follow-up listening feedback selected P05 over the previous P06 direction. A 40-second P05 comparison was generated:

```text
D:\voice-lab\out\mosaic_svc\p0\m1_m2_compare_p05_ittai40_cfg050_steps60\M1_P05_raw_ittai40_lufs.wav
D:\voice-lab\out\mosaic_svc\p0\m1_m2_compare_p05_ittai40_cfg050_steps60\M2_P05_prompt_adapter_ittai40_lufs.wav
D:\voice-lab\out\mosaic_svc\p0\m1_m2_compare_p05_ittai40_cfg050_steps60\metrics.csv
```

```text
M1 P05 raw 40s: f0_corr=0.968344 cent_rmse=92.98 uv_mismatch=0.139872
M2 P05 adapter 40s: f0_corr=0.994381 cent_rmse=84.41 uv_mismatch=0.224898
```

Current decision: keep P05 as canonical, but treat the Prompt Adapter as too strong until a lower-gate variant proves it does not increase UV errors.

Use `--prompt-adapter-strength` at inference time to test lower adapter strength without retraining.

The first lower-strength check used `--prompt-adapter-strength 0.5` and is the current best numeric candidate:

```text
D:\voice-lab\share_audio\mosaic_M2_P05_prompt_adapter_s050_ittai40_cfg050_steps60.mp3
```

```text
M1 P05 raw 40s: f0_corr=0.968344 cent_rmse=92.98 uv_mismatch=0.139872
M2 P05 adapter strength 0.5: f0_corr=0.996016 cent_rmse=48.94 uv_mismatch=0.123041
M2 P05 adapter strength 1.0: f0_corr=0.994381 cent_rmse=84.41 uv_mismatch=0.224898
```

Current decision: use P05 + Prompt Adapter strength 0.5 as the next listening candidate.

Longer prompt check: a 24-second prompt spanning 48s-72s of the same high-quality target song outperformed the 12-second P05 prompt numerically without using an adapter.

```text
D:\voice-lab\share_audio\mosaic_M1_P48_72_24s_prompt_ittai40_cfg050_steps60.mp3
```

```text
M1 P05 12s prompt: f0_corr=0.968344 cent_rmse=92.98 uv_mismatch=0.139872
M1 48-72s 24s prompt: f0_corr=0.996700 cent_rmse=44.93 uv_mismatch=0.073418
```

Current decision after this check: prioritize high-quality prompt mel/semantic selection before adding adapters. The 24-second 48s-72s prompt is the next practical candidate to listen to.

## P0 Commands

Automatically select the prompt that best preserves the current song's F0 while retaining target identity:

```powershell
D:\voice-lab\seed-vc\.venv\Scripts\python.exe -m mosaic_svc.p0.auto_prompt_select `
  --source D:\voice-lab\data\guide_vocals\song.wav `
  --prompt "D:\voice-lab\out\mosaic_svc\p0\prompt_candidates\dadadada_12s\*.wav" `
  --profile D:\voice-lab\out\mosaic_svc\speaker_profiles\target_singing_profile.pt `
  --output D:\voice-lab\out\mosaic_svc\prompt_selection\song
```

The selector extracts a challenging 30-second source probe, renders every prompt at 20 steps, and ranks them using F0 RMSE, F0 correlation, UV retention, and CAMPPlus target identity. It only renders the winner over the original input at 60 steps when the probe passes the default quality gate (`cent_rmse <= 250`, `f0_corr >= 0.85`, `uv_mismatch <= 0.20`). Otherwise it records a rejection and skips the final render. Pass `--render-winner false` to perform selection only.

For source vocals with intermittent RMVPE octave errors, enable the conservative dual-estimator lock:

```powershell
python -m mosaic_svc.p0.infer_p0 `
  --source source_vocal.wav `
  --prompt canonical_prompt.wav `
  --f0-condition True `
  --f0-consensus-lock True
```

The lock changes RMVPE only when a high-confidence pYIN anchor disagrees by an integer octave for a sustained region. It does not snap normal pitch movement or apply a global key shift. A sibling `.f0consensus.json` report records every corrected region.

Audit target clips:

```powershell
D:\voice-lab\seed-vc\.venv\Scripts\python.exe -m mosaic_svc.p0.data_audit `
  --input D:\voice-lab\data\target_clean `
  --output D:\voice-lab\out\mosaic_svc\audit\target_clean.csv
```

Build a prototype bank from an approved CSV manifest:

```powershell
D:\voice-lab\seed-vc\.venv\Scripts\python.exe -m mosaic_svc.p0.build_prototypes `
  --manifest D:\voice-lab\out\mosaic_svc\manifests\canonical_manifest.csv `
  --canonical D:\voice-lab\data\target_clean\dadadada_tenshi_ref_25s.wav `
  --output D:\voice-lab\out\mosaic_svc\p0\prototype_bank.pt
```

Run P0 fixed canonical prompt:

```powershell
D:\voice-lab\seed-vc\.venv\Scripts\python.exe -m mosaic_svc.p0.infer_p0 `
  --source D:\voice-lab\data\guide_vocals\input.wav `
  --prompt D:\voice-lab\data\target_clean\dadadada_tenshi_ref_25s.wav `
  --output D:\voice-lab\out\mosaic_svc\p0 `
  --f0-condition True `
  --diffusion-steps 40
```

Run P0 D0 prototype style correction:

```powershell
D:\voice-lab\seed-vc\.venv\Scripts\python.exe -m mosaic_svc.p0.infer_p0 `
  --source D:\voice-lab\data\guide_vocals\input.wav `
  --prompt D:\voice-lab\data\target_clean\dadadada_tenshi_ref_25s.wav `
  --prototype-bank D:\voice-lab\out\mosaic_svc\p0\prototype_bank.pt `
  --output D:\voice-lab\out\mosaic_svc\p0 `
  --f0-condition True `
  --diffusion-steps 40
```

Train only the Style-Slice Adapter:

```powershell
D:\voice-lab\seed-vc\.venv\Scripts\python.exe -m mosaic_svc.p0.train_style_adapter `
  --manifest D:\voice-lab\out\mosaic_svc\manifests\canonical_manifest.csv `
  --canonical D:\voice-lab\data\target_clean\dadadada_tenshi_ref_25s.wav `
  --output D:\voice-lab\out\mosaic_svc\p0\adapter `
  --steps 300 `
  --initial-scale 0.05
```

Run A/B/C/D0 with LUFS-normalized outputs and metric CSV:

```powershell
D:\voice-lab\seed-vc\.venv\Scripts\python.exe -m mosaic_svc.p0.run_p0_suite `
  --source D:\voice-lab\data\guide_vocals\ittai_itsukara_head_15s.wav `
  --canonical D:\voice-lab\data\target_clean\dadadada_tenshi_ref_25s.wav `
  --style-adapter D:\voice-lab\out\mosaic_svc\p0\adapter\style_adapter_final.pt `
  --prototype-bank D:\voice-lab\out\mosaic_svc\p0\prototype_bank.pt `
  --output D:\voice-lab\out\mosaic_svc\p0\suite_ittai15_steps20 `
  --diffusion-steps 20
```

Run prompt/style path ablation:

```powershell
D:\voice-lab\seed-vc\.venv\Scripts\python.exe -m mosaic_svc.p0.run_path_ablation `
  --source D:\voice-lab\data\guide_vocals\ittai_itsukara_head_15s.wav `
  --prompt-a D:\voice-lab\data\target_clean\dadadada_tenshi_ref_25s.wav `
  --style-a D:\voice-lab\data\target_clean\dadadada_tenshi_ref_25s.wav `
  --prompt-b D:\voice-lab\data\seedvc_refs\maneki_primary_ref.wav `
  --style-b D:\voice-lab\data\seedvc_refs\maneki_primary_ref.wav `
  --output D:\voice-lab\out\mosaic_svc\p0\path_ablation_ittai15_dadadada_vs_maneki_steps20 `
  --diffusion-steps 20
```

Build and sweep prompt candidates:

```powershell
D:\voice-lab\seed-vc\.venv\Scripts\python.exe -m mosaic_svc.p0.build_prompt_candidates `
  --input D:\voice-lab\data\target_clean\dadadada_tenshi_vocal.wav `
  --output-dir D:\voice-lab\out\mosaic_svc\p0\prompt_candidates\dadadada_12s `
  --seconds 12 `
  --hop-seconds 12 `
  --max-candidates 8

D:\voice-lab\seed-vc\.venv\Scripts\python.exe -m mosaic_svc.p0.run_prompt_sweep `
  --source D:\voice-lab\data\guide_vocals\ittai_itsukara_head_15s.wav `
  --manifest D:\voice-lab\out\mosaic_svc\p0\prompt_candidates\dadadada_12s\prompt_candidates.csv `
  --output D:\voice-lab\out\mosaic_svc\p0\prompt_sweep_ittai15_dadadada_12s_steps20 `
  --diffusion-steps 20
```

Build a dialogue speaker profile and rerank high-quality prompt candidates:

```powershell
D:\voice-lab\seed-vc\.venv\Scripts\python.exe -m mosaic_svc.p0.build_speaker_profile `
  --input D:\voice-lab\out\dialogue\maneki_karaoke_stream\maneki_dialogue_strict_denoised.wav `
  --output D:\voice-lab\out\mosaic_svc\speaker_profiles\maneki_dialogue25_campplus.pt `
  --max-segments 96

D:\voice-lab\seed-vc\.venv\Scripts\python.exe -m mosaic_svc.p0.rank_prompts_by_profile `
  --manifest D:\voice-lab\out\mosaic_svc\p0\prompt_candidates\dadadada_12s\prompt_candidates.csv `
  --profile D:\voice-lab\out\mosaic_svc\speaker_profiles\maneki_dialogue25_campplus.pt `
  --output D:\voice-lab\out\mosaic_svc\p0\prompt_candidates\dadadada_12s\prompt_ranked_by_dialogue_profile.csv
```

Train and run the M2 Prompt Adapter:

```powershell
D:\voice-lab\seed-vc\.venv\Scripts\python.exe -m mosaic_svc.p0.train_prompt_adapter `
  --manifest D:\voice-lab\out\mosaic_svc\p0\prompt_candidates\dadadada_12s\prompt_candidates.csv `
  --canonical D:\voice-lab\out\mosaic_svc\p0\prompt_candidates\dadadada_12s\prompt_05_048.00s.wav `
  --output D:\voice-lab\out\mosaic_svc\p0\prompt_adapter_p05_singing_only `
  --steps 300

D:\voice-lab\seed-vc\.venv\Scripts\python.exe -m mosaic_svc.p0.infer_p0 `
  --source D:\voice-lab\data\guide_vocals\ittai_itsukara_head_15s.wav `
  --prompt D:\voice-lab\out\mosaic_svc\p0\prompt_candidates\dadadada_12s\prompt_05_048.00s.wav `
  --prompt-adapter D:\voice-lab\out\mosaic_svc\p0\prompt_adapter_p05_singing_only\prompt_adapter_final.pt `
  --output D:\voice-lab\out\mosaic_svc\p0\m2_p05_adapter_cfg050_steps60 `
  --diffusion-steps 60 `
  --inference-cfg-rate 0.50 `
  --prompt-seconds 12
```

## P0 Comparison IDs

- `A`: upstream Seed-VC zero-shot.
- `B`: fixed canonical prompt, no adapter, no prototype.
- `C`: B plus Style-Slice Adapter.
- `D0`: C plus inference-only CAMPPlus prototype correction.
- `D1`: optional future light training of prototype/gate after D0 passes.
- `M1`: best fixed prompt from prompt sweep/rerank, no adapter.
- `M2`: M1 plus Prompt Adapter.
- `M3`: M2 adapter trained on high-quality singing clips only.
- `M4`: dialogue-derived speaker profile used for prompt selection/reranking, not acoustic training.

## Design Constraint

The 44.1 kHz Seed-VC SVC model uses prompt semantic, prompt mel, and CAMPPlus global style. Mosaic-SVC P0 only adapts the CAMPPlus style path first. It does not add raw mel residuals or acoustic patch retrieval.

## P4-P8 Adaptation Result

The current practical default combines two frozen-base adapters:

- P6: rank-8 K/V-only LoRA on Transformer layers 4, 8, and 12, checkpoint step 600.
- P7: rank-4 CAMPPlus global Style-Slice Adapter, checkpoint step 600.
- P8: both adapters enabled with the fixed P07 canonical prompt.

On three unseen 15-second song clips, the high-quality singing CAMPPlus profile improved from `0.712374` to `0.724428`, while the quality score improved from `0.920236` to `0.925862`.

Train P7 with song-separated validation:

```powershell
python -m mosaic_svc.p7.train_style_slice `
  --train-manifest train_manifest.csv `
  --validation-manifest validation_manifest.csv `
  --canonical canonical_mid.wav `
  --canonical canonical_high.wav `
  --output out\p7 `
  --steps 800
```

Run the combined adaptation:

```powershell
python -m mosaic_svc.p0.infer_p0 `
  --source input.wav `
  --prompt canonical_high.wav `
  --kv-lora out\p6\kv_lora_step_000600.pt `
  --style-adapter out\p7\style_adapter_step_000600.pt `
  --diffusion-steps 60 `
  --inference-cfg-rate 0.50 `
  --seed 1234 `
  --output out\p8
```

Use a high-quality singing profile as the primary singing-identity metric. The low-quality dialogue profile is auxiliary because microphone quality and speech register can reverse adaptation rankings.

## P10 Identity-Aware Adaptation

P10 continues training only the Style-Slice Adapter through frozen BigVGAN and frozen CAMPPlus. The objective is CFM reconstruction plus a small cosine loss against a high-quality singing identity centroid.

```powershell
python -m mosaic_svc.p10.train_identity_aware `
  --train-manifest train_manifest.csv `
  --validation-manifest validation_manifest.csv `
  --canonical canonical_high.wav `
  --style-adapter out\p7\style_adapter_step_000600.pt `
  --kv-lora out\p6\kv_lora_step_000600.pt `
  --identity-profile out\singing_identity.pt `
  --output out\p10 `
  --steps 100
```

On the same three held-out clips, P10 slightly improved identity, quality, UV mismatch, and aggregate rerank over P8 while keeping F0 metrics effectively unchanged. It is accepted under the No-Harm rule without further weight sweeping.
