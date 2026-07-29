# Mosaic-SVC R1.6 Implementation Notes

This fork keeps upstream Seed-VC intact and adds Mosaic-SVC as an extension package.

## Implemented

- P0 frozen Seed-VC inference with fixed canonical prompt.
- P0 prompt/style ablation via separate `--prompt` and `--style-audio`.
- P0 inference-only CAMPPlus prototype correction.
- P0 Style-Slice Adapter rank 4, installed by wrapping `cond_x_merge_linear`.
- P0 Style-Slice Adapter training that updates only the adapter.
- R1.6 data audit and admission CSV generation.
- R1.6 minimal F0/LUFS-proxy evaluation.
- R1.6 streaming module interfaces for Content Student, L1 Prototype Memory, Acoustic Converter, and NSF vocoder stub.
- R1.6 distillation losses for frame, delta, delta2, and phoneme preservation.

## Not Implemented As Production Models Yet

- Full ContentVec + Whisper teacher fusion.
- De-Timbre Adapter pretraining.
- External multi-speaker GRL probe.
- Fully trained Streaming Student.
- Production harmonic-noise NSF vocoder.
- Level 2 mid-block K/V correction.

Those parts now have explicit module boundaries and training contracts, but they still need datasets and training runs.

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

## P0 Comparison IDs

- `A`: upstream Seed-VC zero-shot.
- `B`: fixed canonical prompt, no adapter, no prototype.
- `C`: B plus Style-Slice Adapter.
- `D0`: C plus inference-only CAMPPlus prototype correction.
- `D1`: optional future light training of prototype/gate after D0 passes.

## Design Constraint

The 44.1 kHz Seed-VC SVC model uses prompt semantic, prompt mel, and CAMPPlus global style. Mosaic-SVC P0 only adapts the CAMPPlus style path first. It does not add raw mel residuals or acoustic patch retrieval.
