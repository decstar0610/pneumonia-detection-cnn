# One-shot dataset download (Windows / PowerShell).
# Prereq: `pip install kaggle` and kaggle.json in %USERPROFILE%\.kaggle\  (see data/DOWNLOAD.md).
# For RSNA you must first accept the rules at:
#   https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/rules
#
# Usage (from repo root):
#   .\data\download.ps1            # both datasets
#   .\data\download.ps1 -Only kaggle
#   .\data\download.ps1 -Only rsna

param(
    [ValidateSet("both", "kaggle", "rsna")]
    [string]$Only = "both"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot   # repo root
$kaggleDir = Join-Path $root "data\raw\kaggle_chest_xray"
$rsnaDir   = Join-Path $root "data\raw\rsna"

if ($Only -in @("both", "kaggle")) {
    Write-Host "==> Kaggle Chest X-Ray Pneumonia -> $kaggleDir"
    kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p $kaggleDir --unzip
}

if ($Only -in @("both", "rsna")) {
    Write-Host "==> RSNA Pneumonia Detection Challenge -> $rsnaDir"
    Write-Host "    (must have accepted the competition rules on kaggle.com first)"
    kaggle competitions download -c rsna-pneumonia-detection-challenge -p $rsnaDir
    $zip = Join-Path $rsnaDir "rsna-pneumonia-detection-challenge.zip"
    if (Test-Path $zip) {
        Expand-Archive $zip -DestinationPath $rsnaDir -Force
    }
}

Write-Host "`nDone. Verify with:  python -m src.data"
