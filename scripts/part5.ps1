param([string]$PythonExecutable = "", [switch]$NoPlots)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$tempPath = Join-Path $projectRoot "work\temp"
$cachePath = Join-Path $projectRoot "work\cache"
New-Item -ItemType Directory -Force -Path $tempPath, $cachePath | Out-Null
$env:TEMP = $tempPath
$env:TMP = $tempPath
$env:TMPDIR = $tempPath
$env:PIP_CACHE_DIR = Join-Path $cachePath "pip"
$env:PYTHONPYCACHEPREFIX = Join-Path $cachePath "pycache"
$env:MPLCONFIGDIR = Join-Path $cachePath "matplotlib"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:PYTHONPATH = Join-Path $projectRoot "src"
if (-not $PythonExecutable) {
    $venvPython = Join-Path $projectRoot "work\.venv\Scripts\python.exe"
    $PythonExecutable = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { "python" }
}
& $PythonExecutable -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw "Verification tests failed." }
$arguments = @("-m", "coordinate_geometry.part5_experiments", "--config", "configs/part5.json", "--output", "outputs/part5")
if ($NoPlots) { $arguments += "--no-plots" }
& $PythonExecutable @arguments
if ($LASTEXITCODE -ne 0) { throw "Module 5 verification failed." }
& $PythonExecutable -m coordinate_geometry.verify_part5 --output outputs/part5
if ($LASTEXITCODE -ne 0) { throw "Independent artifact verification failed." }
