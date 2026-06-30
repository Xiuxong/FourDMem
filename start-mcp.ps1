# FourDMem MCP Server — Windows Startup Script
# Usage: .\start-mcp.ps1 [--db-path PATH]
#
# This script starts the FourDMem MCP Server for use with
# oh-my-pi or any MCP-compatible client.

param(
    [string]$DbPath = "$PSScriptRoot\data\vault\evidence.db"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$VenvPython = "$ProjectRoot\python\.venv\Scripts\python.exe"

# ── Pre-flight checks ─────────────────────────────────

if (-not (Test-Path $VenvPython)) {
    Write-Error "Python venv not found at: $VenvPython"
    Write-Host "Run 'make init' or create venv manually:" -ForegroundColor Yellow
    Write-Host "  cd python; python -m venv .venv; .\.venv\Scripts\pip install maturin" -ForegroundColor Yellow
    exit 1
}

# Verify fourdmem module is importable
Write-Host "Verifying fourdmem module..." -ForegroundColor Cyan
& $VenvPython -c "import fourdmem; print('fourdmem OK')" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to import fourdmem. Run 'maturin develop' first."
    exit 1
}

# ── Start MCP Server ──────────────────────────────────

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " FourDMem MCP Server v0.1.0" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host " Database: $DbPath" -ForegroundColor Gray
Write-Host " Python:   $VenvPython" -ForegroundColor Gray
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# mcp_server package lives in python/ — must launch from there
Set-Location "$ProjectRoot\python"
& $VenvPython -m mcp_server.server --db-path $DbPath
