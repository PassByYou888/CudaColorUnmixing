#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$Preset = "cuda",
    [string]$Config = "Release",
    [switch]$Reconfigure
)

$ErrorActionPreference = "Stop"
$Root = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
Push-Location $Root
try {
    if ($Reconfigure) {
        Write-Host "== removing build\$Preset (Reconfigure)" -ForegroundColor Yellow
        Remove-Item -Recurse -Force ".\build\$Preset" -ErrorAction SilentlyContinue
    }

    Write-Host "== cmake --preset $Preset" -ForegroundColor Green
    cmake --preset $Preset
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed (exit $LASTEXITCODE)" }

    Write-Host "== cmake --build --preset $Preset --config $Config" -ForegroundColor Green
    cmake --build --preset $Preset --config $Config
    if ($LASTEXITCODE -ne 0) { throw "cmake build failed (exit $LASTEXITCODE)" }

    Write-Host ""
    Write-Host "== artifacts:" -ForegroundColor Green
    $targets = @(
        @{ Name = "cuda_smoke";     Path = ".\build\$Preset\$Config\cuda_smoke.exe" },
        @{ Name = "bmp_smoke";      Path = ".\build\$Preset\$Config\bmp_smoke.exe" },
        @{ Name = "cpu_smoke";      Path = ".\build\$Preset\$Config\cpu_smoke.exe" },
        @{ Name = "l2_verify";      Path = ".\build\$Preset\$Config\l2_verify.exe" },
        @{ Name = "l3_verify";      Path = ".\build\$Preset\$Config\l3_verify.exe" },
        @{ Name = "l4_verify";      Path = ".\build\$Preset\$Config\l4_verify.exe" },
        @{ Name = "color-unmixing"; Path = ".\build\$Preset\color-unmixing\$Config\color-unmixing.exe" }
    )
    foreach ($t in $targets) {
        if (Test-Path $t.Path) {
            $len = (Get-Item $t.Path).Length
            Write-Host ("  [{0}] {1}  ({2} bytes)" -f $t.Name, $t.Path, $len) -ForegroundColor Cyan
        } else {
            Write-Host ("  [{0}] MISSING: {1}" -f $t.Name, $t.Path) -ForegroundColor Yellow
        }
    }
}
finally {
    Pop-Location
}