param(
  [string]$InputDir = "D:\voice-lab\data\target_clean",
  [string]$OutputCsv = "D:\voice-lab\out\mosaic_svc\audit\target_clean.csv"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
Set-Location $Root
& $Python -m mosaic_svc.r16.data_audit --input $InputDir --output $OutputCsv
