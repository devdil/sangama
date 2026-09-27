param(
  [string]$Version = 'v0.1.0-preview.1',
  [string]$InstallDir = (Join-Path $env:LOCALAPPDATA 'Sangama\bin'),
  [string]$ReleaseDir = ''
)
$ErrorActionPreference = 'Stop'
if ($Version -notmatch '^v[0-9][A-Za-z0-9._-]*$') { throw 'Invalid version' }
if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { throw 'This release supports native Windows x64 only' }
if (-not [IO.Path]::IsPathRooted($InstallDir)) { throw 'InstallDir must be absolute' }
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$temp = Join-Path ([IO.Path]::GetTempPath()) ('sangama-' + [Guid]::NewGuid())
New-Item -ItemType Directory -Path $temp | Out-Null
try {
  $asset = "sangama-$Version-x86_64-pc-windows-msvc.zip"
  foreach ($name in @($asset, 'SHA256SUMS')) {
    $dest = Join-Path $temp $name
    if ($ReleaseDir) { Copy-Item -LiteralPath (Join-Path $ReleaseDir $name) -Destination $dest }
    else { Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/devdil/sangama/releases/download/$Version/$name" -OutFile $dest }
  }
  $lines = @(Get-Content (Join-Path $temp 'SHA256SUMS') | Where-Object { $_ -match ('^[0-9a-f]{64}  ' + [Regex]::Escape($asset) + '$') })
  if ($lines.Count -ne 1) { throw 'Missing or ambiguous checksum' }
  $actual = (Get-FileHash (Join-Path $temp $asset) -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($actual -ne $lines[0].Substring(0,64)) { throw 'Checksum mismatch; nothing installed' }
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  $zip = [IO.Compression.ZipFile]::OpenRead((Join-Path $temp $asset))
  try {
    $entry = $zip.GetEntry('sangama.exe')
    if ($null -eq $entry) { throw 'Executable missing' }
    [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, (Join-Path $temp 'sangama.exe'))
  } finally { $zip.Dispose() }
  New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
  Copy-Item -LiteralPath (Join-Path $temp 'sangama.exe') -Destination (Join-Path $InstallDir 'sangama.exe') -Force
  & (Join-Path $InstallDir 'sangama.exe') --version
  if ($LASTEXITCODE -ne 0) { throw 'Executable failed; verify platform/runtime support' }
  Write-Host "Installed to $InstallDir. PATH was not modified."
  Write-Host 'Worker setup: https://github.com/devdil/sangama/blob/main/docs/distribution.md'
} finally { Remove-Item -LiteralPath $temp -Recurse -Force }
