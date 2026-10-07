$ErrorActionPreference = "Stop"

$url = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
if (-not (Test-Path "model.gguf")) {
  Invoke-WebRequest -Uri $url -OutFile model.gguf -UseBasicParsing
}

$probe = "target/x86_64-pc-windows-msvc/release/examples/tool_probe.exe"
if (-not (Test-Path $probe)) { throw "probe not built at $probe" }
$out = & $probe model.gguf 2>&1 | Out-String
if ($LASTEXITCODE -ne 0) { throw "Windows inference probe exited $LASTEXITCODE" }
Write-Host $out

if ($out -notmatch 'tool_format detected: Llama3Json') {
  throw "tool format not detected on Windows"
}
if ($out -notmatch 'parser fixture : true') {
  throw "deterministic tool parser fixture failed"
}

$buffered = [regex]::Match($out, 'tokens\s+:\s+(\d+) prompt / (\d+) completion')
$streamed = [regex]::Match($out, 'stream usage\s+:\s+Some\(\((\d+), (\d+)\)\)')
if (-not $buffered.Success -or -not $streamed.Success) {
  throw "could not read token counts from the probe"
}
if ($buffered.Groups[1].Value -ne $streamed.Groups[1].Value) {
  throw "prompt tokens differ between paths: $($buffered.Groups[1].Value) vs $($streamed.Groups[1].Value)"
}
if ($buffered.Groups[2].Value -ne $streamed.Groups[2].Value) {
  throw "completion tokens differ between paths: $($buffered.Groups[2].Value) vs $($streamed.Groups[2].Value)"
}
if ($out -notmatch 'stream matches: true') {
  throw "streaming and non-streaming inference produced different content"
}

$content = [regex]::Match($out, 'content_len\s+:\s+(\d+)')
if (-not $content.Success -or [int]$content.Groups[1].Value -lt 1) {
  throw "real Windows inference produced no content"
}
$completionTokens = [int]$buffered.Groups[2].Value
if ($completionTokens -lt 1 -or $completionTokens -ge 80) {
  throw "completion tokens $completionTokens hit the cap - real inference did not finish"
}

Write-Host "OK: Windows real inference matches across both paths ($($buffered.Groups[1].Value)/$completionTokens tokens); deterministic tool parser fixture passed"
