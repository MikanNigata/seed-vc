param(
  [Parameter(Mandatory=$true)][string]$ManifestCsv,
  [string]$Canonical = "D:\voice-lab\data\target_clean\dadadada_tenshi_ref_25s.wav",
  [string]$OutputDir = "D:\voice-lab\out\mosaic_svc\p0\adapter",
  [int]$Steps = 300
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
Set-Location $Root
& $Python -m mosaic_svc.p0.train_style_adapter `
  --manifest $ManifestCsv `
  --canonical $Canonical `
  --output $OutputDir `
  --steps $Steps `
  --fp16 True
