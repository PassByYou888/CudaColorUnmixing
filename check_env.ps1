#Requires -Version 5.1
<#
.SYNOPSIS
    CUDA 12.6 + VS2022 环境自检 (for ColorUnmixing CUDA port)

.DESCRIPTION
    检查项:
      - 系统 / PowerShell 版本
      - NVIDIA GPU (名称, 驱动, 显存, 计算能力 Compute Capability)
      - CUDA Toolkit (nvcc 版本, 支持的 GPU 架构)
      - Visual Studio 2022 + MSVC 工具集
      - CUDA 的 VS Build Customizations 集成
      - CMake
      - 第三方库 (Eigen / NLopt) 提示
      - nvcc 编译自测 (可选, 验证 double atomicAdd 可用)
    输出:
      - 控制台彩色报告
      - cuda_env_report.json

.PARAMETER ReportPath
    报告输出路径。默认脚本同目录下 cuda_env_report.json

.PARAMETER ExtraLibRoots
    额外的第三方库搜索根目录 (分号分隔)。例如 "D:\libs;C:\vcpkg\installed\x64-windows"

.PARAMETER SkipCompileTest
    跳过 nvcc 编译自测

.EXAMPLE
    .\check_env.ps1
    .\check_env.ps1 -ExtraLibRoots "C:\vcpkg\installed\x64-windows" -SkipCompileTest
#>
[CmdletBinding()]
param(
    [string]$ReportPath = '',
    [string]$ExtraLibRoots = '',
    [switch]$SkipCompileTest
)

$ErrorActionPreference = 'Continue'

# ---------- 状态容器 ----------
$script:Checks   = New-Object System.Collections.Generic.List[object]
$script:Warnings = New-Object System.Collections.Generic.List[string]
$script:Failures = New-Object System.Collections.Generic.List[string]

function Add-Check {
    param(
        [string]$Category,
        [string]$Name,
        [ValidateSet('OK','WARN','FAIL','INFO')]
        [string]$Status,
        [string]$Detail,
        $Data = $null
    )
    $script:Checks.Add([pscustomobject]@{
        Category = $Category
        Name     = $Name
        Status   = $Status
        Detail   = $Detail
        Data     = $Data
    })
    switch ($Status) {
        'WARN' { $script:Warnings.Add("[$Category] $Name : $Detail") }
        'FAIL' { $script:Failures.Add("[$Category] $Name : $Detail") }
    }
}

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor DarkCyan
    Write-Host "  $Title" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor DarkCyan
}

# ---------- 辅助 ----------
function Get-FirstExistingPath {
    param([string[]]$Candidates)
    foreach ($p in $Candidates) {
        if ($p -and (Test-Path $p)) { return $p }
    }
    return $null
}

function Invoke-Capture {
    param(
        [string]$Exe,
        [string[]]$Arguments
    )
    try {
        $out = & $Exe @Arguments 2>&1 | Out-String
        return [pscustomobject]@{ Ok = ($LASTEXITCODE -eq 0); Output = $out; ExitCode = $LASTEXITCODE }
    } catch {
        return [pscustomobject]@{ Ok = $false; Output = $_.Exception.Message; ExitCode = -1 }
    }
}

function Resolve-CCFromName {
    param([string]$Name)
    if (-not $Name) { return $null }
    # 注意: 顺序按 "越新越具体" 排
    $map = @(
        @{ P = 'RTX\s*50\d0|5090|5080|5070|5060';                   CC = '12.0' },
        @{ P = '\bB200\b|\bB100\b';                                 CC = '10.0' },
        @{ P = '\bH100\b|\bH200\b|\bH800\b';                        CC = '9.0'  },
        @{ P = 'RTX\s*40\d0|\b4090\b|\b4080\b|\b4070\b|\b4060\b';   CC = '8.9'  },
        @{ P = '\bL40S\b|\bL40\b|\bL4\b|\bA10G\b|\bA10\b|\bA40\b|\bA30\b'; CC = '8.6' },
        @{ P = 'RTX\s*30\d0|\b3090\b|\b3080\b|\b3070\b|\b3060\b';   CC = '8.6'  },
        @{ P = '\bA100\b';                                          CC = '8.0'  },
        @{ P = 'RTX\s*20\d0|\b2080\b|\b2070\b|\b2060\b|Titan\s*RTX';CC = '7.5'  },
        @{ P = 'GTX\s*16\d0|\b1650\b|\b1660\b';                     CC = '7.5'  },
        @{ P = 'Titan\s*V|\bV100\b';                                CC = '7.0'  },
        @{ P = 'GTX\s*10\d0|\b1080\b|\b1070\b|\b1060\b|\b1050\b';   CC = '6.1'  }
    )
    foreach ($e in $map) {
        if ($Name -match $e.P) { return $e.CC }
    }
    return $null
}

function Convert-CCToArch {
    param([string]$CC)
    if ([string]::IsNullOrWhiteSpace($CC)) { return $null }
    return ($CC -replace '\.', '').Trim()
}

# ============================================================
# 1. 系统
# ============================================================
Write-Section "1. 系统信息"

$osDesc = try { (Get-CimInstance Win32_OperatingSystem -ErrorAction Stop).Caption } catch { $null }
$psVer  = $PSVersionTable.PSVersion.ToString()
$is64   = [Environment]::Is64BitOperatingSystem

if ($osDesc) {
    Add-Check 'System' 'OS' 'OK' $osDesc
} else {
    Add-Check 'System' 'OS' 'WARN' '无法读取 OS 信息'
}
Add-Check 'System' 'PowerShell' 'OK' $psVer
Add-Check 'System' '64-bit OS' $(if ($is64) {'OK'} else {'FAIL'}) $is64

# ============================================================
# 2. GPU
# ============================================================
Write-Section "2. NVIDIA GPU"

$gpus = @()
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) {
    Add-Check 'GPU' 'nvidia-smi' 'FAIL' '未找到 nvidia-smi，请安装 NVIDIA 驱动'
} else {
    # 优先查询 compute_cap (较新驱动支持)
    $q = Invoke-Capture -Exe $smi.Source -Arguments @(
        '--query-gpu=index,name,driver_version,memory.total,compute_cap',
        '--format=csv,noheader,nounits'
    )
    if ($q.Ok -and $q.Output -match ',') {
        foreach ($line in ($q.Output -split "`r?`n")) {
            if ([string]::IsNullOrWhiteSpace($line)) { continue }
            $parts = $line -split ',\s*'
            if ($parts.Count -ge 5) {
                $gpus += [pscustomobject]@{
                    Index      = [int]$parts[0]
                    Name       = $parts[1].Trim()
                    Driver     = $parts[2].Trim()
                    MemoryMB   = [int](($parts[3] -replace '[^\d]', ''))
                    ComputeCap = $parts[4].Trim()
                }
            }
        }
    }
    # 回退: 无 compute_cap
    if ($gpus.Count -eq 0) {
        $q2 = Invoke-Capture -Exe $smi.Source -Arguments @(
            '--query-gpu=index,name,driver_version,memory.total',
            '--format=csv,noheader,nounits'
        )
        if ($q2.Ok) {
            foreach ($line in ($q2.Output -split "`r?`n")) {
                if ([string]::IsNullOrWhiteSpace($line)) { continue }
                $parts = $line -split ',\s*'
                if ($parts.Count -ge 4) {
                    $nm = $parts[1].Trim()
                    $gpus += [pscustomobject]@{
                        Index      = [int]$parts[0]
                        Name       = $nm
                        Driver     = $parts[2].Trim()
                        MemoryMB   = [int](($parts[3] -replace '[^\d]', ''))
                        ComputeCap = (Resolve-CCFromName $nm)
                    }
                }
            }
        }
    }

    if ($gpus.Count -eq 0) {
        Add-Check 'GPU' 'nvidia-smi query' 'FAIL' 'nvidia-smi 可执行但无 GPU 信息'
    } else {
        foreach ($g in $gpus) {
            $ccStr = if ($g.ComputeCap) { $g.ComputeCap } else { '未知' }
            $memGb = [math]::Round($g.MemoryMB / 1024.0, 1)
            Add-Check 'GPU' ("GPU #" + $g.Index) 'OK' `
                ("{0} | Driver {1} | {2} GB | CC {3}" -f $g.Name, $g.Driver, $memGb, $ccStr) `
                $g

            # 驱动版本 vs CUDA 12.6 最低要求 (>= 560.x on Windows)
            if ($g.Driver -match '^(\d+)\.') {
                $maj = [int]$Matches[1]
                if ($maj -lt 560) {
                    Add-Check 'GPU' ("Driver for #" + $g.Index) 'WARN' `
                        ("{0} < 560，CUDA 12.6 官方要求 >= 560.76 (Windows)；可能触发向后兼容或运行时失败" -f $g.Driver)
                } else {
                    Add-Check 'GPU' ("Driver for #" + $g.Index) 'OK' ("{0} >= 560" -f $g.Driver)
                }
            }

            # double atomicAdd 要求 sm_60+
            if ($g.ComputeCap) {
                $archNum = [int](Convert-CCToArch $g.ComputeCap)
                if ($archNum -ge 60) {
                    Add-Check 'GPU' ("sm_" + $archNum + " >= sm_60") 'OK' '支持原生 double atomicAdd'
                } else {
                    Add-Check 'GPU' ("sm_" + $archNum + " < sm_60") 'FAIL' `
                        '不支持原生 double atomicAdd，bins 原子累加需 CAS 实现，性能与精度均受影响'
                }
            }
        }
    }
}

# ============================================================
# 3. CUDA Toolkit
# ============================================================
Write-Section "3. CUDA Toolkit"

$cudaInfo = $null
$nvccCmd = Get-Command nvcc -ErrorAction SilentlyContinue
$nvccPath = if ($nvccCmd) { $nvccCmd.Source } else { $null }
if (-not $nvccPath -and $env:CUDA_PATH) {
    $cand = Join-Path $env:CUDA_PATH 'bin\nvcc.exe'
    if (Test-Path $cand) { $nvccPath = $cand }
}
# 直接扫 Program Files 下的 CUDA 目录，找到 12.6
if (-not $nvccPath) {
    $cand = Get-ChildItem 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA' -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending | Select-Object -First 1
    if ($cand) {
        $candNvcc = Join-Path $cand.FullName 'bin\nvcc.exe'
        if (Test-Path $candNvcc) { $nvccPath = $candNvcc }
    }
}

if (-not $nvccPath) {
    Add-Check 'CUDA' 'nvcc' 'FAIL' '未找到 nvcc.exe，请确认 CUDA Toolkit 已安装'
} else {
    Add-Check 'CUDA' 'nvcc path' 'OK' $nvccPath

    $ver = Invoke-Capture -Exe $nvccPath -Arguments @('--version')
    if ($ver.Ok) {
        $release = $null; $tkv = $null
        if ($ver.Output -match 'release\s+([\d\.]+)') { $release = $Matches[1] }
        if ($ver.Output -match 'V([\d\.]+)')        { $tkv     = $Matches[1] }

        if ($release -eq '12.6') {
            Add-Check 'CUDA' 'Toolkit version' 'OK' ("release {0} ({1})" -f $release, $tkv)
        } elseif ($release -match '^12\.') {
            Add-Check 'CUDA' 'Toolkit version' 'WARN' ("release {0}，目标为 12.6，注意兼容性" -f $release)
        } else {
            Add-Check 'CUDA' 'Toolkit version' 'FAIL' ("release {0}，目标为 12.6" -f $release)
        }

        # 支持的 GPU 架构
        $listArch = Invoke-Capture -Exe $nvccPath -Arguments @('--list-gpu-arch')
        $supportedArchs = @()
        if ($listArch.Ok) {
            foreach ($line in ($listArch.Output -split "`r?`n")) {
                if ($line -match 'compute_(\d+)') { $supportedArchs += $Matches[1] }
            }
        }
        $supportedArchs = $supportedArchs | Select-Object -Unique
        Add-Check 'CUDA' 'Supported GPU archs' 'INFO' ($supportedArchs -join ', ')
        $cudaInfo = [pscustomobject]@{
            NvccPath        = $nvccPath
            Release         = $release
            ToolkitVersion  = $tkv
            SupportedArchs  = $supportedArchs
        }
    } else {
        Add-Check 'CUDA' 'nvcc --version' 'FAIL' $ver.Output
    }
}

if ($env:CUDA_PATH) {
    Add-Check 'CUDA' 'CUDA_PATH' 'OK' $env:CUDA_PATH
} else {
    Add-Check 'CUDA' 'CUDA_PATH' 'WARN' '未设置 CUDA_PATH 环境变量'
}

# ============================================================
# 4. Visual Studio 2022
# ============================================================
Write-Section "4. Visual Studio 2022"

$vsInfo = $null
$vswhereCandidates = @(
    (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'),
    'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
)
$vswhere = Get-FirstExistingPath $vswhereCandidates

if (-not $vswhere) {
    Add-Check 'VS' 'vswhere' 'FAIL' '未找到 vswhere.exe，可能未安装 VS 或 Installer 损坏'
} else {
    $vsRes = Invoke-Capture -Exe $vswhere -Arguments @(
        '-latest', '-products', '*',
        '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
        '-property', 'installationPath'
    )
    $vsPath = if ($vsRes.Ok) { $vsRes.Output.Trim() } else { '' }
    if ([string]::IsNullOrWhiteSpace($vsPath) -or -not (Test-Path $vsPath)) {
        Add-Check 'VS' 'VS2022 C++ tools' 'FAIL' '未找到含 C++ 工具集的 VS 安装'
    } else {
        Add-Check 'VS' 'VS2022 path' 'OK' $vsPath

        # 检查是否为 VS2022 (v170)
        $is2022 = $vsPath -match '\\2022\\'
        Add-Check 'VS' 'Version 2022' $(if ($is2022) {'OK'} else {'WARN'}) $vsPath

        # MSVC 版本
        $msvcRoot = Join-Path $vsPath 'VC\Tools\MSVC'
        if (Test-Path $msvcRoot) {
            $versions = Get-ChildItem $msvcRoot -Directory | Sort-Object Name -Descending
            $clPath = $null; $msvcVer = $null
            foreach ($v in $versions) {
                $c = Join-Path $v.FullName 'bin\Hostx64\x64\cl.exe'
                if (Test-Path $c) { $clPath = $c; $msvcVer = $v.Name; break }
            }
            if ($clPath) {
                Add-Check 'VS' 'MSVC toolset' 'OK' ("{0} => {1}" -f $msvcVer, $clPath)
            } else {
                Add-Check 'VS' 'MSVC toolset' 'FAIL' '未找到 Hostx64\x64\cl.exe'
            }
        } else {
            Add-Check 'VS' 'MSVC toolset' 'FAIL' "$msvcRoot 不存在"
        }

        # CUDA Build Customizations (VS 集成)
        $bcDirs = @(
            (Join-Path $vsPath 'MSBuild\Microsoft\VC\v170\BuildCustomizations'),
            (Join-Path $vsPath 'VC\BuildCustomizations')
        )
        $cudaProps = @()
        foreach ($d in $bcDirs) {
            if (Test-Path $d) {
                $cudaProps += Get-ChildItem $d -Filter 'CUDA *.props' -ErrorAction SilentlyContinue |
                              Select-Object -ExpandProperty Name
            }
        }
        $cudaProps = $cudaProps | Select-Object -Unique
        if ($cudaProps.Count -gt 0) {
            $has126 = $cudaProps -match 'CUDA\s+12\.6\.props'
            Add-Check 'VS' 'CUDA Build Customizations' $(if ($has126) {'OK'} else {'WARN'}) `
                ($cudaProps -join '; ')
        } else {
            Add-Check 'VS' 'CUDA Build Customizations' 'FAIL' `
                '未找到 CUDA *.props，VS 中无法直接编译 .cu。建议重装 CUDA 并勾选 Visual Studio Integration，或改用 CMake+CUDA 语言模式'
        }

        $vsInfo = [pscustomobject]@{
            InstallPath = $vsPath
            MsvcVersion = if ($msvcVer) { $msvcVer } else { $null }
            ClPath      = if ($clPath) { $clPath } else { $null }
            CudaProps   = $cudaProps
        }
    }
}

# ============================================================
# 5. CMake
# ============================================================
Write-Section "5. CMake"

$cmakeInfo = $null
$cmakeCmd = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmakeCmd) {
    Add-Check 'CMake' 'cmake' 'FAIL' '未找到 cmake，请安装 CMake 3.25+ 并加入 PATH'
} else {
    $v = Invoke-Capture -Exe $cmakeCmd.Source -Arguments @('--version')
    $ver = if ($v.Ok -and $v.Output -match 'cmake version\s+([\d\.]+)') { $Matches[1] } else { $null }
    if ($ver) {
        $majorMinor = ($ver -split '\.')[0..1] -join '.'
        $ok = [version]($majorMinor + '.0') -ge [version]'3.25.0'
        Add-Check 'CMake' 'cmake' $(if ($ok) {'OK'} else {'WARN'}) `
            ("{0} ({1})；CUDA 语言模式建议 >= 3.25" -f $ver, $cmakeCmd.Source)
        $cmakeInfo = [pscustomobject]@{ Path = $cmakeCmd.Source; Version = $ver }
    } else {
        Add-Check 'CMake' 'cmake' 'FAIL' $v.Output
    }
}

# 额外工具
foreach ($tool in @('ninja', 'git')) {
    $c = Get-Command $tool -ErrorAction SilentlyContinue
    if ($c) {
        Add-Check 'Tools' $tool 'OK' $c.Source
    } else {
        Add-Check 'Tools' $tool 'INFO' '未安装 (可选)'
    }
}

# ============================================================
# 6. 第三方库 (Eigen / NLopt) 提示
# ============================================================
Write-Section "6. 第三方库 (Eigen / NLopt)"

$libRoots = @()
if ($ExtraLibRoots) {
    $libRoots += ($ExtraLibRoots -split ';' | Where-Object { $_ })
}
# 环境变量
if ($env:EIGEN3_INCLUDE_DIR) { $libRoots += (Split-Path $env:EIGEN3_INCLUDE_DIR -Parent) }
if ($env:EIGEN_ROOT)        { $libRoots += $env:EIGEN_ROOT }
if ($env:NLOPT_ROOT)        { $libRoots += $env:NLOPT_ROOT }
if ($env:VCPKG_ROOT)        { $libRoots += (Join-Path $env:VCPKG_ROOT 'installed\x64-windows') }

# vcpkg 检测
$vcpkg = Get-Command vcpkg -ErrorAction SilentlyContinue
if ($vcpkg) {
    Add-Check 'Libs' 'vcpkg' 'OK' $vcpkg.Source
} else {
    Add-Check 'Libs' 'vcpkg' 'INFO' '未安装 (可选；推荐用于装 Eigen/NLopt)'
}

function Find-Header {
    param([string]$HeaderRelPath, [string[]]$Roots)
    foreach ($r in $Roots) {
        if (-not $r) { continue }
        foreach ($sub in @('', 'include', 'include\eigen3', 'include\nlopt')) {
            $p = Join-Path $r (Join-Path $sub $HeaderRelPath)
            if (Test-Path $p) { return $p }
        }
    }
    return $null
}

$eigenHit = Find-Header 'Eigen\Core' $libRoots
$nloptHit = Find-Header 'nlopt.hpp' $libRoots
if ($eigenHit) {
    Add-Check 'Libs' 'Eigen' 'OK' $eigenHit
} else {
    Add-Check 'Libs' 'Eigen' 'WARN' '未在常见路径找到 Eigen/Core。可用 vcpkg install eigen3:x64-windows'
}
if ($nloptHit) {
    Add-Check 'Libs' 'NLopt' 'OK' $nloptHit
} else {
    Add-Check 'Libs' 'NLopt' 'WARN' '未在常见路径找到 nlopt.hpp。可用 vcpkg install nlopt:x64-windows'
}
Add-Check 'Libs' 'dlib (仅完整 zAI 需要)' 'INFO' 'ColorUnmixing 复刻本身不需要 dlib；完整 zAI 库需要'

# ============================================================
# 7. nvcc 编译自测 (可选)
# ============================================================
if (-not $SkipCompileTest -and $cudaInfo -and $gpus.Count -gt 0) {
    Write-Section "7. nvcc 编译自测"

    $archForTest = $null
    if ($gpus[0].ComputeCap) { $archForTest = Convert-CCToArch $gpus[0].ComputeCap }
    if (-not $archForTest -or [int]$archForTest -lt 60) { $archForTest = '75' }

    $tmp = Join-Path $env:TEMP ("cuda_check_" + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $tmp -Force | Out-Null
    $src = Join-Path $tmp 'test.cu'
    $obj = Join-Path $tmp 'test.obj'
    $errFile = Join-Path $tmp 'err.txt'
    $outFile = Join-Path $tmp 'out.txt'

    @'
#include <cstdio>
__global__ void k(double* p) { atomicAdd(p, 1.0); }
int main() { return 0; }
'@ | Set-Content -Path $src -Encoding ASCII

    try {
        $args = @('-c', "`"$src`"", '-o', "`"$obj`"", "-arch=sm_$archForTest")
        $proc = Start-Process -FilePath $cudaInfo.NvccPath -ArgumentList $args `
                 -Wait -PassThru -NoNewWindow `
                 -RedirectStandardError $errFile -RedirectStandardOutput $outFile

        $errText = if (Test-Path $errFile) { (Get-Content $errFile -Raw) } else { '' }
        if ($proc.ExitCode -eq 0 -and (Test-Path $obj)) {
            Add-Check 'CUDA' ('compile test (sm_' + $archForTest + ')') 'OK' `
                'nvcc 编译通过，double atomicAdd 可用'
        } else {
            Add-Check 'CUDA' ('compile test (sm_' + $archForTest + ')') 'FAIL' `
                ("ExitCode={0}; stderr: {1}" -f $proc.ExitCode, ($errText -split "`r?`n" | Select-Object -First 3) -join ' ')
        }
    } catch {
        Add-Check 'CUDA' 'compile test' 'FAIL' $_.Exception.Message
    } finally {
        Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# ============================================================
# 8. 汇总
# ============================================================
Write-Section "检查结果汇总"
$script:Checks | Format-Table -AutoSize Category, Name, Status, Detail | Out-String -Width 200 | Write-Host

if ($script:Failures.Count -gt 0) {
    Write-Host "[FAILURES]" -ForegroundColor Red
    $script:Failures | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    Write-Host ""
}
if ($script:Warnings.Count -gt 0) {
    Write-Host "[WARNINGS]" -ForegroundColor Yellow
    $script:Warnings | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
    Write-Host ""
}

# ---- 关键结论 ----
Write-Section "关键结论 / 推荐参数"

$recArchList = @()
foreach ($g in $gpus) {
    if ($g.ComputeCap) {
        $a = Convert-CCToArch $g.ComputeCap
        if ([int]$a -ge 60 -and $a -notin $recArchList) { $recArchList += $a }
    }
}
$archFallback = $false
if ($recArchList.Count -eq 0) {
    $recArchList = @('75')
    $archFallback = $true
}
$archArg = ($recArchList -join ';')

Write-Host "CUDA 版本检查:        $(if ($cudaInfo -and $cudaInfo.Release -eq '12.6') {'[OK] 12.6'} elseif ($cudaInfo) {'[WARN] ' + $cudaInfo.Release} else {'[FAIL] 未找到'})"
Write-Host "VS2022 检查:          $(if ($vsInfo) {'[OK] ' + $vsInfo.InstallPath} else {'[FAIL]'})"
Write-Host "GPU 数量:             $($gpus.Count)"
foreach ($g in $gpus) {
    Write-Host ("  - #{0} {1} | CC {2} | {3} GB" -f $g.Index, $g.Name, $(if ($g.ComputeCap) { $g.ComputeCap } else { '?' }), [math]::Round($g.MemoryMB / 1024.0, 1))
}
Write-Host "推荐 CMAKE_CUDA_ARCHITECTURES: $archArg $(if ($archFallback) {'(fallback，请手动确认)'} else {''})" -ForegroundColor Green
Write-Host ""
Write-Host "下一步示例 (CMake 配置命令):" -ForegroundColor Cyan
Write-Host @"
  cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ``
    -DCMAKE_CUDA_ARCHITECTURES=$archArg ``
    -DUSE_CUDA=ON ``
    -DEigen3_DIR="<你的 Eigen3Config.cmake 路径>" ``
    -DNLopt_DIR="<你的 NLoptConfig.cmake 路径>"
"@ -ForegroundColor Gray

# ---- 报告 ----
if (-not $ReportPath) {
    $base = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
    $ReportPath = Join-Path $base 'cuda_env_report.json'
}

$report = [ordered]@{
    Timestamp             = (Get-Date).ToString('s')
    Host                  = $env:COMPUTERNAME
    User                  = $env:USERNAME
    OS                    = $osDesc
    PowerShell            = $psVer
    Is64BitOS             = $is64
    CUDA                  = $cudaInfo
    GPUs                  = $gpus
    VisualStudio          = $vsInfo
    CMake                 = $cmakeInfo
    RecommendedCudaArch   = $archArg
    Checks                = $script:Checks
    Warnings              = $script:Warnings
    Failures              = $script:Failures
}
try {
    $report | ConvertTo-Json -Depth 8 | Set-Content -Path $ReportPath -Encoding UTF8
    Write-Host ""
    Write-Host "报告已写入: $ReportPath" -ForegroundColor Green
} catch {
    Write-Host "报告写入失败: $($_.Exception.Message)" -ForegroundColor Red
}

# 退出码: 0=全 OK/WARN, 1=有 FAIL
if ($script:Failures.Count -gt 0) { exit 1 } else { exit 0 }