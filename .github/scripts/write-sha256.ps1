param(
  [Parameter(Mandatory = $true)][string]$Path,
  [Parameter(Mandatory = $true)][string]$Output,
  [Parameter(Mandatory = $true)][string]$RecordedName
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path $Path)) { throw "checksum input not found: $Path" }
$hash = (Get-FileHash -Algorithm SHA256 $Path).Hash.ToLowerInvariant()
$line = "$hash  $RecordedName`n"
# Use an explicit LF so the manifests work with POSIX sha256sum --check during
# draft assembly. PowerShell's default text writer uses CRLF, which becomes part
# of the filename when coreutils verifies the manifest on Ubuntu.
[System.IO.File]::WriteAllText($Output, $line, [System.Text.Encoding]::ASCII)
Write-Host $line.TrimEnd()
