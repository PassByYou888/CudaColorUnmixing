#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$Preset = "cpu",
    [string]$Exe    = "cpu_smoke",
    [int]   $Width  = 64,
    [int]   $Height = 64,
    [string]$OutDir = "out",
    [string]$Config = "Release",
    [switch]$Clean,
    [switch]$NoRun
)

$ErrorActionPreference = "Stop"
$Root = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
Push-Location $Root
try {
    Write-Host "== workspace: $Root" -ForegroundColor Cyan
    Write-Host "== preset:    $Preset" -ForegroundColor Cyan
    Write-Host "== config:    $Config" -ForegroundColor Cyan
    Write-Host "== exe:       $Exe" -ForegroundColor Cyan
    Write-Host "== output:    $OutDir" -ForegroundColor Cyan
    Write-Host ""

    if ($Clean) {
        Write-Host "== cleaning build/$Preset and $OutDir" -ForegroundColor Yellow
        Remove-Item -Recurse -Force ".\build\$Preset" -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force ".\$OutDir"      -ErrorAction SilentlyContinue
    }

    if (-not (Test-Path $OutDir)) {
        New-Item -ItemType Directory -Path $OutDir | Out-Null
    }

    Write-Host "== cmake --preset $Preset" -ForegroundColor Green
    cmake --preset $Preset
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed (exit $LASTEXITCODE)" }

    Write-Host "== cmake --build --preset $Preset --config $Config" -ForegroundColor Green
    cmake --build --preset $Preset --config $Config
    if ($LASTEXITCODE -ne 0) { throw "cmake build failed (exit $LASTEXITCODE)" }

    $exePath = Join-Path $Root "build\$Preset\$Config\$Exe.exe"
    if (-not (Test-Path $exePath)) {
        throw "executable not found: $exePath (check -Exe / -Config)"
    }

    if ($NoRun) {
        Write-Host ""
        Write-Host "== built: $exePath (skipping run, -NoRun)" -ForegroundColor Green
        return
    }

    Write-Host ""
    Write-Host "== running: $exePath $Width $Height $OutDir" -ForegroundColor Green
    & $exePath $Width $Height $OutDir
    if ($LASTEXITCODE -ne 0) { throw "$Exe failed (exit $LASTEXITCODE)" }

    Write-Host ""
    Write-Host "== outputs in ${OutDir}:" -ForegroundColor Green
    Get-ChildItem (Join-Path $Root $OutDir) -Filter *.bmp -ErrorAction SilentlyContinue |
        Sort-Object Name |
        Select-Object Name, Length |
        Format-Table -AutoSize |
        Out-String -Width 200 |
        Write-Host
}
finally {
    Pop-Location
}