param(
  [string]$InputAudio = "D:\voice-lab\data\target_clean\dadadada_tenshi_vocal.wav",
  [string]$OutputDir = "D:\voice-lab\out\mosaic_svc\p0\prompt_candidates\dadadada",
  [double]$Seconds = 12.0,
  [double]$HopSeconds = 10.0,
  [int]$MaxCandidates = 8
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
Set-Location $Root
& $Python -m mosaic_svc.p0.build_prompt_candidates `
  --input $InputAudio `
  --output-dir $OutputDir `
  --seconds $Seconds `
  --hop-seconds $HopSeconds `
  --max-candidates $MaxCandidates
