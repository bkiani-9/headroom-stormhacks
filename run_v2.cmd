@echo off
rem Headroom, second interface: the whole grid as a live one-line diagram.
rem Double-click this file, or run .\run_v2.cmd in PowerShell from this folder.
rem It opens on http://localhost:8502, so app.py can keep running on 8501.
cd /d "%~dp0"
py -3.14 -m streamlit run app_v2.py --server.port 8502 --theme.base dark --theme.backgroundColor "#0A0A0A" --theme.secondaryBackgroundColor "#1A1A1A" --theme.textColor "#FFFFFF" --theme.primaryColor "#FFFFFF" --theme.borderColor "#555555" --theme.font "Segoe UI Variable Text, Segoe UI, sans-serif" --theme.baseFontSize 16 --theme.baseRadius "10px"
