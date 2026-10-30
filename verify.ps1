#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$Preset  = "cpu",
    [string]$Exe     = "cpu_smoke",
    [int]   $Width   = 64,
    [int]   $Height  = 64,
    [string]$OutDir  = "out",
    [string]$RefDir  = "out_ref",
    [string]$Config  = "Release",
    [switch]$Establish
)

$ErrorActionPreference = "Stop"
$Root = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
Push-Location $Root
try {
    if ($Establish) {
        Write-Host "== establishing reference: $RefDir" -ForegroundColor Yellow
        Remove-Item -Recurse -Force ".\$RefDir" -ErrorAction SilentlyContinue
        New-Item -ItemType Directory -Path $RefDir | Out-Null
    }

    Write-Host "== running $Exe ($Preset, $Config)" -ForegroundColor Cyan
    .\run_cpu.ps1 -Preset $Preset -Exe $Exe -Width $Width -Height $Height -OutDir $OutDir -Config $Config
    if ($LASTEXITCODE -ne 0) { throw "run_cpu.ps1 failed" }

    if ($Establish) {
        Write-Host "== copying $OutDir -> $RefDir (flat)" -ForegroundColor Yellow
        Get-ChildItem (Join-Path $Root $OutDir) -Filter *.bmp | ForEach-Object {
            Copy-Item -Path $_.FullName -Destination (Join-Path $RefDir $_.Name) -Force
        }
        Write-Host "== reference established, $((Get-ChildItem (Join-Path $Root $RefDir) -Filter *.bmp).Count) files" -ForegroundColor Green
        return
    }

    if (-not (Test-Path $RefDir)) {
        throw "reference dir $RefDir not found — run .\verify.ps1 -Establish first"
    }

    Write-Host "== comparing $OutDir vs $RefDir (SHA256)" -ForegroundColor Cyan
    $diff = 0
    $missing = 0
    $files = Get-ChildItem (Join-Path $Root $OutDir) -Filter *.bmp | Sort-Object Name
    foreach ($f in $files) {
        $ref = Join-Path $RefDir $f.Name
        if (-not (Test-Path $ref)) {
            Write-Host "  MISSING in ref: $($f.Name)" -ForegroundColor Red
            $missing++
            continue
        }
        $h1 = (Get-FileHash $f.FullName -Algorithm SHA256).Hash
        $h2 = (Get-FileHash $ref -Algorithm SHA256).Hash
        if ($h1 -ne $h2) {
            Write-Host "  DIFF: $($f.Name)" -ForegroundColor Red
            $diff++
        }
    }
    $refCount = (Get-ChildItem (Join-Path $Root $RefDir) -Filter *.bmp).Count
    if ($refCount -ne $files.Count) {
        Write-Host "  COUNT MISMATCH: out=$($files.Count) ref=$refCount" -ForegroundColor Red
    }
    Write-Host ""
    if ($diff -eq 0 -and $missing -eq 0 -and $refCount -eq $files.Count) {
        Write-Host "== PASS: $($files.Count) files, all SHA256 match" -ForegroundColor Green
    } else {
        Write-Host "== FAIL: diff=$diff missing=$missing ref=$refCount out=$($files.Count)" -ForegroundColor Red
        exit 1
    }
}
finally {
    Pop-Location
}