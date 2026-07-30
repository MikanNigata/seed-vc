param(
  [Parameter(Mandatory=$true)][string]$Source,
  [string]$Manifest = "D:\voice-lab\out\mosaic_svc\p0\prompt_candidates\dadadada\prompt_candidates.csv",
  [string]$OutputDir = "D:\voice-lab\out\mosaic_svc\p0\prompt_sweep",
  [int]$DiffusionSteps = 20
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
Set-Location $Root
& $Python -m mosaic_svc.p0.run_prompt_sweep `
  --source $Source `
  --manifest $Manifest `
  --output $OutputDir `
  --diffusion-steps $DiffusionSteps
