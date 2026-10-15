param([string]$PythonExecutable = "")
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
$env:PYTHONPATH = Join-Path $projectRoot "src"
if (-not $PythonExecutable) {
    $venvPython = Join-Path $projectRoot "work\.venv\Scripts\python.exe"
    $PythonExecutable = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { "python" }
}
& $PythonExecutable -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw "Mathematical checks failed." }
& $PythonExecutable -m coordinate_geometry.smoke --config configs/smoke.json --output outputs/smoke
if ($LASTEXITCODE -ne 0) { throw "Smoke run failed." }
