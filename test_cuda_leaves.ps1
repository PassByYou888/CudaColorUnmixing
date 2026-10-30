#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$InputImage = "leaves.bmp",
    [string]$OutDir     = "leaves_cuda",
    [string]$Backend    = "cuda",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$Root = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
Push-Location $Root
try {
    if (-not $SkipBuild) {
        & "$Root\build.ps1" -Preset cuda -Config Release
    }

    $exe = ".\build\cuda\Release\bmp_smoke.exe"
    if (-not (Test-Path $exe))      { throw "executable not found: $exe" }
    if (-not (Test-Path $InputImage)) { throw "input not found: $InputImage" }

    Remove-Item -Recurse -Force ".\$OutDir" -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Path $OutDir | Out-Null

    Write-Host ""
    Write-Host "== running: $exe $InputImage $OutDir $Backend" -ForegroundColor Green
    & $exe $InputImage $OutDir $Backend
    if ($LASTEXITCODE -ne 0) { throw "bmp_smoke failed (exit $LASTEXITCODE)" }

    Write-Host ""
    Write-Host "== outputs in ${OutDir}:" -ForegroundColor Green
    Get-ChildItem ".\$OutDir" -Filter *.bmp | Sort-Object Name |
        Select-Object Name, Length |
        Format-Table -AutoSize |
        Out-String -Width 200 |
        Write-Host
}
finally {
    Pop-Location
}