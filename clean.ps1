#Requires -Version 5.1
[CmdletBinding()]
param(
    [switch]$Build,
    [switch]$All
)

$ErrorActionPreference = "Stop"
$Root = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
Push-Location $Root
try {
    $outputDirs = @(
        "leaves",
        "leaves_cuda",
        "out",
        "out_cuda",
        "out_cuda_256",
        "out_cuda_512",
        "out_cuda_1024",
        "out_ref",
        "out_ref_cuda",
        "out_before_2_2_1",
        "nude"
    )

    $buildDirs = @(
        "build\cpu",
        "build\cuda",
        "build\vulkan",
        "build\all"
    )

    Write-Host "== cleaning output directories" -ForegroundColor Yellow
    foreach ($d in $outputDirs) {
        if (Test-Path $d) {
            Remove-Item -Recurse -Force $d
            Write-Host "  removed: $d"
        }
    }

    if ($Build -or $All) {
        Write-Host "== cleaning build directories" -ForegroundColor Yellow
        foreach ($d in $buildDirs) {
            if (Test-Path $d) {
                Remove-Item -Recurse -Force $d
                Write-Host "  removed: $d"
            }
        }
    }

    Write-Host "== done" -ForegroundColor Green
}
finally {
    Pop-Location
}