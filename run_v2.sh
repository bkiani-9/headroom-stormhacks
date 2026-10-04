#!/bin/sh
# Headroom, second interface, for Mac and Linux (run_v2.cmd is the Windows
# version). Run it from this folder with:  sh run_v2.sh
# It opens on http://localhost:8502.
cd "$(dirname "$0")"
PY=python3
command -v python3.14 >/dev/null 2>&1 && PY=python3.14
"$PY" -m streamlit run app_v2.py --server.port 8502 --theme.base dark \
  --theme.backgroundColor "#0A0A0A" --theme.secondaryBackgroundColor "#1A1A1A" \
  --theme.textColor "#FFFFFF" --theme.primaryColor "#FFFFFF" --theme.borderColor "#555555" \
  --theme.font "Segoe UI Variable Text, Segoe UI, sans-serif" \
  --theme.baseFontSize 16 --theme.baseRadius "10px"
