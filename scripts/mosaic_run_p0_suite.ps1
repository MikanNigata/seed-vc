param(
  [Parameter(Mandatory=$true)][string]$Source,
  [string]$Canonical = "D:\voice-lab\data\target_clean\dadadada_tenshi_ref_25s.wav",
  [string]$OutputDir = "D:\voice-lab\out\mosaic_svc\p0\suite",
  [string]$StyleAdapter = "",
  [string]$PrototypeBank = "",
  [int]$DiffusionSteps = 30
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$ArgsList = @(
  "-m", "mosaic_svc.p0.run_p0_suite",
  "--source", $Source,
  "--canonical", $Canonical,
  "--output", $OutputDir,
  "--diffusion-steps", "$DiffusionSteps"
)
if ($StyleAdapter -ne "") { $ArgsList += @("--style-adapter", $StyleAdapter) }
if ($PrototypeBank -ne "") { $ArgsList += @("--prototype-bank", $PrototypeBank) }
Set-Location $Root
& $Python @ArgsList
