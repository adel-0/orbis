# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. Always read .env for config values instead of config.py.

## Common Development Commands

### Running the Application
```bash
# Always activate virtual environment first
source .venv/bin/activate  # Bash
.\.venv\Scripts\Activate.ps1  # PowerShell

# Start the API server
uv run main.py
# Server runs on http://localhost:7887 by default
```

### Dependencies
```bash
# Install all dependencies
uv sync
```

### Folders to ignore
Always exclude .venv and __pycache__ folders when analysing folders.

## Implementation Lessons Learned

Core principle: KISS.