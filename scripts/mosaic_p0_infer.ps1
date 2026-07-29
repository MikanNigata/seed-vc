param(
  [Parameter(Mandatory=$true)][string]$Source,
  [string]$Prompt = "D:\voice-lab\data\target_clean\dadadada_tenshi_ref_25s.wav",
  [string]$OutputDir = "D:\voice-lab\out\mosaic_svc\p0",
  [string]$StyleAudio = "",
  [string]$StyleAdapter = "",
  [string]$PrototypeBank = "",
  [int]$DiffusionSteps = 40
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$ArgsList = @(
  "-m", "mosaic_svc.p0.infer_p0",
  "--source", $Source,
  "--prompt", $Prompt,
  "--output", $OutputDir,
  "--f0-condition", "True",
  "--fp16", "True",
  "--diffusion-steps", "$DiffusionSteps"
)
if ($StyleAudio -ne "") { $ArgsList += @("--style-audio", $StyleAudio) }
if ($StyleAdapter -ne "") { $ArgsList += @("--style-adapter", $StyleAdapter) }
if ($PrototypeBank -ne "") { $ArgsList += @("--prototype-bank", $PrototypeBank) }
Set-Location $Root
& $Python @ArgsList
