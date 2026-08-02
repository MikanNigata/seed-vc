param(
    [Parameter(Mandatory=$true)][string]$DatasetDir,
    [Parameter(Mandatory=$true)][string]$IdentityProfile,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [int]$StudentSteps = 2000,
    [int]$ConverterSteps = 3000,
    [int]$ApSteps = 1500,
    [int]$NsfSteps = 4000
)

$ErrorActionPreference = "Stop"
$Python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Python venv not found: $Python" }

$Train = Join-Path $DatasetDir "train.csv"
$Validation = Join-Path $DatasetDir "validation.csv"
foreach ($Path in @($Train, $Validation, $IdentityProfile)) {
    if (-not (Test-Path $Path)) { throw "Required input not found: $Path" }
}

$P13 = Join-Path $OutputDir "p13_student"
$P14 = Join-Path $OutputDir "p14_converter"
$P15Ap = Join-Path $OutputDir "p15_ap"
$P15Nsf = Join-Path $OutputDir "p15_nsf"

& $Python -m mosaic_svc.p13.train_student --train-manifest $Train --validation-manifest $Validation --output $P13 --steps $StudentSteps
if ($LASTEXITCODE -ne 0) { throw "P13 student training failed with exit code $LASTEXITCODE" }

$Student = Join-Path $P13 "content_student_best.pt"
& $Python -m mosaic_svc.p14.train_converter --train-manifest $Train --validation-manifest $Validation --identity-profile $IdentityProfile --student $Student --output $P14 --steps $ConverterSteps
if ($LASTEXITCODE -ne 0) { throw "P14 converter training failed with exit code $LASTEXITCODE" }

$Converter = Join-Path $P14 "streaming_converter_best.pt"
& $Python -m mosaic_svc.p15.train_ap --train-manifest $Train --validation-manifest $Validation --identity-profile $IdentityProfile --converter $Converter --output $P15Ap --steps $ApSteps
if ($LASTEXITCODE -ne 0) { throw "P15 AP training failed with exit code $LASTEXITCODE" }

& $Python -m mosaic_svc.p15.train_nsf --train-manifest $Train --validation-manifest $Validation --output $P15Nsf --steps $NsfSteps
if ($LASTEXITCODE -ne 0) { throw "P15 NSF training failed with exit code $LASTEXITCODE" }

Write-Host "R1.6 training completed: $OutputDir"
