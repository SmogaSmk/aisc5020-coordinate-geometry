param([string]$PythonExecutable = "python")
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
$venvPath = Join-Path $projectRoot "work\.venv"
if (-not (Test-Path -LiteralPath (Join-Path $venvPath "Scripts\python.exe"))) {
    & $PythonExecutable -m venv $venvPath
    if ($LASTEXITCODE -ne 0) { throw "Creating the project virtual environment failed." }
}
$venvPython = Join-Path $venvPath "Scripts\python.exe"
& $venvPython -m pip install -e .
if ($LASTEXITCODE -ne 0) { throw "Installing project dependencies failed." }
Write-Host "Ready. Run: .\scripts\check.ps1"
