# refresh-wsl-portproxy-8080.ps1
# Requires: Administrator

param(
  [int]$ListenPort = 8080,
  [int]$ConnectPort = 8080,
  [string]$ListenAddress = "0.0.0.0",
  [string]$WslDistro = ""  # e.g. "Ubuntu-22.04", empty = default
)

function Get-WslIPv4 {
  param([string]$Distro)

  $cmd = if ([string]::IsNullOrWhiteSpace($Distro)) {
    "wsl hostname -I"
  } else {
    "wsl -d $Distro hostname -I"
  }

  $out = & powershell -NoProfile -Command $cmd 2>$null
  if (-not $out) { return $null }

  $ips = ($out -split '\s+') | Where-Object { $_ -match '^\d{1,3}(\.\d{1,3}){3}$' }
  if (-not $ips -or $ips.Count -eq 0) { return $null }

  return $ips[0]
}

$wslIp = Get-WslIPv4 -Distro $WslDistro
if (-not $wslIp) {
  Write-Host "[ERR] Failed to get WSL IPv4. Is WSL running?" -ForegroundColor Red
  exit 1
}

Write-Host "[OK] WSL IP = $wslIp"

# Delete old mapping (ignore errors)
& netsh interface portproxy delete v4tov4 listenaddress=$ListenAddress listenport=$ListenPort | Out-Null

# Add new mapping
& netsh interface portproxy add v4tov4 `
  listenaddress=$ListenAddress listenport=$ListenPort `
  connectaddress=$wslIp connectport=$ConnectPort | Out-Null

Write-Host "[OK] portproxy refreshed: $ListenAddress`:$ListenPort -> $wslIp`:$ConnectPort"

# Ensure firewall rule exists
$ruleName = "wsl-portproxy-$ListenPort"
$existing = & netsh advfirewall firewall show rule name="$ruleName" 2>$null
if ($LASTEXITCODE -ne 0 -or -not $existing) {
  & netsh advfirewall firewall add rule name="$ruleName" dir=in action=allow protocol=TCP localport=$ListenPort | Out-Null
  Write-Host "[OK] Firewall rule added: $ruleName"
} else {
  Write-Host "[OK] Firewall rule exists: $ruleName"
}

# Show current mappings for the port
Write-Host "`nCurrent v4tov4 mappings:"
& netsh interface portproxy show v4tov4 | Select-String ":$ListenPort\s"