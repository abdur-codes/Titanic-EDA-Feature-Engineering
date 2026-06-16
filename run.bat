@echo off
cd /d "%~dp0"
echo Opening VS Code for Data Science Project 1...
echo.
echo Once VS Code opens, press Ctrl+Shift+B to run the pipeline
echo.
start "" code "%~dp0"
