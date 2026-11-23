# Local overnight val of the public haideptry fork (np-examples, classes 1/2/3).
# Does not submit. Log: logs\fork_val_npx.log
# Prevents Windows sleep while running; close the window to stop.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Py = (Get-Command python).Source
$Vendor = Join-Path $Root "vendor\haideptry-casmi26"
$LogDir = Join-Path $Root "logs"
$Cache = Join-Path $Root "data\cache"
$Log = Join-Path $LogDir "fork_val_npx.log"
New-Item -ItemType Directory -Force -Path $LogDir, $Cache | Out-Null

$esType = Add-Type -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);' -Name ExecState -Namespace Native -PassThru
[void]$esType::SetThreadExecutionState([uint32]'0x80000001')  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

Write-Host "python: $Py"
Write-Host "log:    $Log"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONPATH = Join-Path $Vendor "src"

$argsList = @(
  (Join-Path $Vendor "scripts\run_fork_val.py"),
  "--data",   (Join-Path $Root "data"),
  "--public", (Join-Path $Root "data\public"),
  "--cache",  $Cache,
  "--split",  "npx",
  "--ranker", (Join-Path $Vendor "artifacts\ranker_versions\v4\rank_train.npz"),
  "--prunes", "400:lib",
  "--classes", "1,2,3",
  "--sim-powers", "4"
)

& $Py -u @argsList *>&1 | Tee-Object -FilePath $Log
$code = $LASTEXITCODE
[void]$esType::SetThreadExecutionState([uint32]'0x80000000')
exit $code
