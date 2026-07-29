param(
  [Parameter(Mandatory=$true)][string]$ManifestCsv,
  [string]$Canonical = "D:\voice-lab\data\target_clean\dadadada_tenshi_ref_25s.wav",
  [string]$OutputBank = "D:\voice-lab\out\mosaic_svc\p0\prototype_bank.pt",
  [double]$MinQualityScore = 0.60
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
Set-Location $Root
& $Python -m mosaic_svc.p0.build_prototypes `
  --manifest $ManifestCsv `
  --canonical $Canonical `
  --output $OutputBank `
  --min-quality-score $MinQualityScore
