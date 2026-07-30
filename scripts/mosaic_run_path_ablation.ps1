param(
  [Parameter(Mandatory=$true)][string]$Source,
  [string]$PromptA = "D:\voice-lab\data\target_clean\dadadada_tenshi_ref_25s.wav",
  [string]$StyleA = "D:\voice-lab\data\target_clean\dadadada_tenshi_ref_25s.wav",
  [string]$PromptB = "D:\voice-lab\data\seedvc_refs\maneki_primary_ref.wav",
  [string]$StyleB = "D:\voice-lab\data\seedvc_refs\maneki_primary_ref.wav",
  [string]$OutputDir = "D:\voice-lab\out\mosaic_svc\p0\path_ablation",
  [int]$DiffusionSteps = 20
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
Set-Location $Root
& $Python -m mosaic_svc.p0.run_path_ablation `
  --source $Source `
  --prompt-a $PromptA `
  --style-a $StyleA `
  --prompt-b $PromptB `
  --style-b $StyleB `
  --output $OutputDir `
  --diffusion-steps $DiffusionSteps
