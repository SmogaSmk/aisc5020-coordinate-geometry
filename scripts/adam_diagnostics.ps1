param([string]$PythonExecutable = "", [switch]$Plot)
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
& $PythonExecutable -m unittest tests.test_adam_diagnostics tests.test_geometry -v
if ($LASTEXITCODE -ne 0) { throw "Module 2 diagnostic tests failed." }
$arguments = @(
    "-m", "coordinate_geometry.adam_diagnostics",
    "--config", "configs/adam_diagnostics.json",
    "--output", "outputs/adam_diagnostics"
)
if ($Plot) { $arguments += "--plot" }
& $PythonExecutable @arguments
if ($LASTEXITCODE -ne 0) { throw "Module 2 diagnostics failed." }
