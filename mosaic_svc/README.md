# Mosaic-SVC R1.6 Implementation Notes

This fork keeps upstream Seed-VC intact and adds Mosaic-SVC as an extension package.

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
- R1.6 data audit and admission CSV generation.
- R1.6 minimal F0/LUFS-proxy evaluation.
- P11 ContentVec + Whisper gated teacher fusion and bounded De-Timbre Adapter.
- P11 timbre-perturbation training and warmup GRL pretraining for external multi-speaker data.
- P12 path-by-path linear/MLP speaker leakage probes.
- P13 causal Content Student with dynamic chunks and multi-loss distillation.
- P14 explicit F0/UV/confidence/slope/energy/phonation bus and causal acoustic converter.
- P15 causal target AP Head and trainable harmonic-noise NSF vocoder with persistent phase.
- P16 file renderer, Gradio GUI, and queued live microphone runtime.
- One-pass dataset preparation and sequential R1.6 training scripts.

## Implementation Versus Checkpoints

The R1.6 module and training paths are implemented. Production P13-P16 checkpoints have not yet been trained on the complete approved singing dataset. The included smoke checkpoints use synthetic data and only prove execution, serialization, CUDA gradients, and runtime wiring; they are not listenable models.

Level 2 K/V correction is already implemented in P6 and remains conditional. Level 3 mel/spectral residual retrieval remains intentionally excluded by design.

## R1.6 End-to-End

Prepare a CSV with `path,split,session`. Every row must explicitly use `train`, `validation`, or `test`; split by song/session, not neighboring segments.

Train P11 on approved target singing:

```powershell
python -m mosaic_svc.p11.train_detimbre `
  --train-manifest train_audio.csv `
  --validation-manifest validation_audio.csv `
  --contentvec D:\voice-lab\models\contentvec-hf `
  --output D:\voice-lab\out\mosaic_svc\r16\p11
```

Prepare all P13-P15 features in one encoder pass per clip:

```powershell
python -m mosaic_svc.p14.prepare_dataset `
  --manifest dataset_split.csv `
  --teacher D:\voice-lab\out\mosaic_svc\r16\p11\content_teacher_best.pt `
  --contentvec D:\voice-lab\models\contentvec-hf `
  --output D:\voice-lab\out\mosaic_svc\r16\dataset
```

Train Student, Converter, AP, then NSF in the required frozen-stage order:

```powershell
.\scripts\mosaic_train_r16.ps1 `
  -DatasetDir D:\voice-lab\out\mosaic_svc\r16\dataset `
  -IdentityProfile D:\voice-lab\out\mosaic_svc\speaker_profiles\singing_identity.pt `
  -OutputDir D:\voice-lab\out\mosaic_svc\r16\models
```

Render a file or launch the GUI:

```powershell
python -m mosaic_svc.p16.infer_file --input source.wav --output converted.wav `
  --student models\p13_student\content_student_best.pt `
  --converter models\p14_converter\streaming_converter_best.pt `
  --ap-head models\p15_ap\ap_head_best.pt `
  --nsf models\p15_nsf\streaming_nsf_best.pt `
  --identity-profile singing_identity.pt --mode render

python -m mosaic_svc.p16.app --student <student.pt> --converter <converter.pt> `
  --ap-head <ap.pt> --nsf <nsf.pt> --identity-profile <identity.pt>
```

Live microphone mode uses a worker queue so GPU inference never runs in the audio callback:

```powershell
python -m mosaic_svc.p16.live_audio --student <student.pt> --converter <converter.pt> `
  --ap-head <ap.pt> --nsf <nsf.pt> --identity-profile <identity.pt> --mode live-quality
```

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

Audit target clips:

```powershell
D:\voice-lab\seed-vc\.venv\Scripts\python.exe -m mosaic_svc.r16.data_audit `
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
